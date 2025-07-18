import io
import os
import shutil
import tempfile
from pathlib import Path
import cv2
import datasets
import pickle
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset as torch_dataset

# Use existing /opt/publaynet directory for all caches to avoid permission issues
cache_base_dir = Path("/opt/publaynet/cache")
os.environ['HF_HOME'] = str(cache_base_dir / 'huggingface')
os.environ['HF_DATASETS_CACHE'] = str(cache_base_dir / 'huggingface' / 'datasets')
os.environ['TRANSFORMERS_CACHE'] = str(cache_base_dir / 'huggingface' / 'transformers')
os.environ['TORCH_HOME'] = str(cache_base_dir / 'torch')
os.environ['TMPDIR'] = str(cache_base_dir / 'tmp')

# Create necessary directories within the existing /opt/publaynet structure
cache_dirs = [
    cache_base_dir / 'huggingface',
    cache_base_dir / 'huggingface' / 'datasets', 
    cache_base_dir / 'huggingface' / 'transformers',
    cache_base_dir / 'torch',
    cache_base_dir / 'tmp'
]

for cache_dir in cache_dirs:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory: {cache_dir}")
    except PermissionError:
        print(f"Warning: Could not create cache directory {cache_dir}, using default caching")

# --- Configuration for Publaynet ---
publaynet_dataset_root_dir = Path("/home/docanalysis/florence2-training/publaynet")
word_pickles_base_dir = Path("/opt/publaynet/Publaynet/VGT_publaynet_grid_pkl")
output_storage_base_dir = Path("/opt/publaynet/Publaynet")
output_dataset_name = "dataset_florence2_publaynet_simple"

# Chunking configuration
CHUNK_SIZE = 50  # Process 50 items at a time - adjust based on your memory/space
MAX_MEMORY_ITEMS = 100  # Keep max 100 items in memory before writing

id2label = {
    1: "text",
    2: "title", 
    3: "list",
    4: "table",
    5: "figure",
}
# --- End Configuration ---

features = Features(
    {
        "image_path": Value("string"),  # Store path instead of actual image
        "category_ids": Sequence(Value("int32")),
        "image_id": Value("int32"),
        "boxes": Sequence(Sequence(Value("float32"))),
        "width": Value("int32"),
        "height": Value("int32"),
        "bbox_str": Value("string"),
        "class_strs": Sequence(Value("string")),
        "ocr_words": Sequence(Value("string")),
        "ocr_boxes": Sequence(Sequence(Value("float32"))),
        "input_ids": Sequence(Value("int32")),
        "subword_bboxs": Sequence(Sequence(Value("float32"))),
    }
)

def quant_bbox(width, height, boxes, category_ids):
    bins_w, bins_h = [1000, 1000]
    size_per_bin_w = width / bins_w
    size_per_bin_h = height / bins_h

    bbox_str_dict_ordered = {cat_id: id2label[cat_id] for cat_id in sorted(list(set(category_ids)))}
    
    temp_cat_bboxes = {cat_id: [] for cat_id in category_ids}
    for bbox, cat_id in zip(boxes, category_ids):
        temp_cat_bboxes[cat_id].append(bbox)

    for cat_id in bbox_str_dict_ordered.keys(): 
        cat_bboxes = temp_cat_bboxes.get(cat_id, [])
        for bbox in cat_bboxes:
            bbox_copy = bbox.copy()
            xmin, ymin, xmax, ymax = torch.tensor(bbox_copy).split(1, dim=-1)
            quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

            quantized_boxes_tensor = torch.cat(
                (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
            ).int()
            if quantized_boxes_tensor.ndim > 1:
                quantized_boxes_tensor = quantized_boxes_tensor.squeeze()

            bbox_str_dict_ordered[cat_id] += (
                f"<loc_{quantized_boxes_tensor[0]}><loc_{quantized_boxes_tensor[1]}><loc_{quantized_boxes_tensor[2]}><loc_{quantized_boxes_tensor[3]}>"
            )

    full_bbox_str = ""
    class_strs = []
    for cat_id_key in sorted(bbox_str_dict_ordered.keys()):
        if len(bbox_str_dict_ordered[cat_id_key]) > len(id2label[cat_id_key]):
             full_bbox_str += bbox_str_dict_ordered[cat_id_key]
             class_strs.append(bbox_str_dict_ordered[cat_id_key])
    return full_bbox_str, class_strs


class COCODataset(torch_dataset):
    def __init__(self, annotation_file: str, image_root_dir: str, word_pickles_split_dir: str, image_size=(1025, 1025)):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_root_dir) 
        self.word_pickles_dir = Path(word_pickles_split_dir)

        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.word_pickles_dir.exists():
            raise FileNotFoundError(f"Word pickles directory not found: {self.word_pickles_dir}")

        self.coco = COCO(str(self.annotation_file))
        image_ids_from_coco = self.coco.getImgIds()
        ids_to_remove = self.filter_img_ids(image_ids_from_coco)
        self.image_ids = [img_id for img_id in image_ids_from_coco if img_id not in ids_to_remove]

    def filter_img_ids(self, img_ids):
        ids_to_remove = set()
        for img_id in img_ids:
            img_info_list = self.coco.loadImgs(img_id)
            if not img_info_list:
                ids_to_remove.add(img_id)
                continue
            img_info = img_info_list[0]
            
            img_path = self.image_dir / img_info["file_name"]
            if not img_path.exists():
                ids_to_remove.add(img_id)
                continue

            base_file_name = Path(img_info["file_name"]).stem 
            pickle_file_name = base_file_name + ".pdf.pkl"
            word_pickle_path_check = self.word_pickles_dir / pickle_file_name
            if not word_pickle_path_check.exists():
                ids_to_remove.add(img_id)
        return list(ids_to_remove)
        
    def __len__(self):
        return len(self.image_ids)

    def _load_word_pickle(self, image_info):
        base_file_name = Path(image_info["file_name"]).stem
        pickle_file_name = base_file_name + ".pdf.pkl"
        word_pickle_file_path = self.word_pickles_dir / pickle_file_name
        with open(word_pickle_file_path, "rb") as f:
            word_info = pickle.load(f)
        return word_info

    def __getitem__(self, idx):
        if idx >= len(self.image_ids):
             raise IndexError(f"Index {idx} is out of bounds for image_ids of length {len(self.image_ids)}")
        
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id] 
        
        image_path = self.image_dir / image_info["file_name"]
        
        # Check if image exists but don't load it - just store the path
        if not image_path.exists():
            return None 

        # Get image dimensions without fully loading the image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            return None

        try:
            word_info = self._load_word_pickle(image_info)
        except FileNotFoundError:
            return None

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
        annotations = [ann for ann in annotations if ann["category_id"] in id2label]

        boxes_coco = [ann["bbox"] for ann in annotations] 
        category_ids = [ann["category_id"] for ann in annotations]
        
        boxes_xyxy = [
            [box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes_coco
        ]
        
        bbox_str, class_strs = quant_bbox(width, height, boxes_xyxy, category_ids)

        output = {
            "image_path": str(image_path),  # Store path instead of image
            "category_ids": category_ids,
            "image_id": image_id, 
            "boxes": boxes_xyxy,
            "width": width,
            "height": height,
            "bbox_str": bbox_str,
            "class_strs": class_strs,
            "ocr_words": word_info["texts"],
            "ocr_boxes": word_info["bbox_texts_list"],
            "input_ids": word_info["input_ids"],
            "subword_bboxs": word_info["bbox_subword_list"],
        }
        
        return output


def save_chunk_to_disk(chunk_data, chunk_dir, chunk_num):
    """Save a chunk of data as a small dataset"""
    if not chunk_data:
        return False
    
    try:
        chunk_dataset = Dataset.from_list(chunk_data, features=features)
        chunk_path = chunk_dir / f"chunk_{chunk_num:06d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        
        # Save with minimal processes to reduce memory usage
        chunk_dataset.save_to_disk(str(chunk_path), num_proc=1)
        print(f"    Saved chunk {chunk_num} with {len(chunk_data)} items to {chunk_path}")
        
        # Clear memory
        del chunk_dataset, chunk_data
        
        return True
    except Exception as e:
        print(f"    Error saving chunk {chunk_num}: {e}")
        return False


def combine_chunks_to_final_dataset(chunks_dir, final_output_dir):
    """Combine all chunks into final dataset"""
    print(f"Combining chunks from {chunks_dir} into final dataset at {final_output_dir}")
    
    chunk_paths = sorted([p for p in chunks_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")])
    
    if not chunk_paths:
        print("No chunks found to combine!")
        return False
    
    print(f"Found {len(chunk_paths)} chunks to combine")
    
    try:
        # Load first chunk as base
        combined_dataset = Dataset.load_from_disk(str(chunk_paths[0]))
        print(f"Loaded base chunk with {len(combined_dataset)} items")
        
        # Concatenate remaining chunks
        for i, chunk_path in enumerate(chunk_paths[1:], 1):
            print(f"Loading and combining chunk {i+1}/{len(chunk_paths)}: {chunk_path.name}")
            chunk_dataset = Dataset.load_from_disk(str(chunk_path))
            
            # Use concatenate_datasets to combine
            from datasets import concatenate_datasets
            combined_dataset = concatenate_datasets([combined_dataset, chunk_dataset])
            
            # Clean up
            del chunk_dataset
            
            if (i + 1) % 10 == 0:  # Progress update every 10 chunks
                print(f"    Combined {i+1} chunks so far, total items: {len(combined_dataset)}")
        
        print(f"Final combined dataset has {len(combined_dataset)} items")
        
        # Save final dataset
        shutil.rmtree(final_output_dir, ignore_errors=True)
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        combined_dataset.save_to_disk(str(final_output_dir), num_proc=1)
        print(f"Successfully saved final dataset to {final_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error combining chunks: {e}")
        return False


def process_split_in_chunks(coco_dataset_instance, dataset_type, output_dir):
    """Process a dataset split in chunks to manage memory and storage"""
    
    num_images = len(coco_dataset_instance)
    print(f"Processing {num_images} images for '{dataset_type}' in chunks of {CHUNK_SIZE}")
    
    # Create temporary directory for chunks
    chunks_dir = output_dir / f"{dataset_type}_chunks"
    shutil.rmtree(chunks_dir, ignore_errors=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    current_chunk = []
    chunk_num = 0
    total_processed = 0
    
    for i in range(num_images):
        if (i + 1) % 100 == 0:
            print(f"  Processing item {i+1}/{num_images} for {dataset_type}...")
        
        try:
            item_data = coco_dataset_instance[i]
            if item_data:
                current_chunk.append(item_data)
                total_processed += 1
                
                # Check if chunk is full or if we've reached memory limit
                if len(current_chunk) >= CHUNK_SIZE:
                    # Save current chunk
                    if save_chunk_to_disk(current_chunk, chunks_dir, chunk_num):
                        chunk_num += 1
                    
                    # Clear chunk and force garbage collection
                    current_chunk = []
                    import gc
                    gc.collect()
                    
        except Exception as e:
            print(f"Error processing item {i} for {dataset_type}: {e}")
            continue
    
    # Save remaining items in final chunk
    if current_chunk:
        save_chunk_to_disk(current_chunk, chunks_dir, chunk_num)
        chunk_num += 1
    
    print(f"Finished chunking for '{dataset_type}': {total_processed} items in {chunk_num} chunks")
    
    # Now combine all chunks into final dataset
    final_split_dir = output_dir / dataset_type
    success = combine_chunks_to_final_dataset(chunks_dir, final_split_dir)
    
    if success:
        # Clean up chunk files to save space
        print(f"Cleaning up temporary chunks for '{dataset_type}'...")
        shutil.rmtree(chunks_dir, ignore_errors=True)
        print(f"Successfully created final dataset for '{dataset_type}' with {total_processed} items")
    
    return success


if __name__ == "__main__":
    dataset_types_to_process = ["train", "val"] 
    final_output_root = output_storage_base_dir / output_dataset_name
    final_output_root.mkdir(parents=True, exist_ok=True)
    print(f"Processed dataset will be saved under: {final_output_root}")
    print(f"Using chunk size: {CHUNK_SIZE} items per chunk")
    print(f"Cache directories redirected to /opt to avoid user space issues")

    for dataset_type in dataset_types_to_process:
        print(f"\n--- Processing '{dataset_type}' split ---")

        current_images_path = publaynet_dataset_root_dir / dataset_type
        current_coco_json_path = publaynet_dataset_root_dir / f"{dataset_type}.json"

        if not current_coco_json_path.exists():
            print(f"Annotation file {current_coco_json_path} not found. Skipping '{dataset_type}' split.")
            continue
        if not current_images_path.exists():
            print(f"Image directory {current_images_path} not found. Skipping '{dataset_type}' split.")
            continue

        pickle_split_folder_name = "dev" if dataset_type == "val" else dataset_type
        current_word_pickles_path = word_pickles_base_dir / pickle_split_folder_name
        
        if not current_word_pickles_path.exists():
            print(f"Word pickle directory {current_word_pickles_path} not found. Skipping '{dataset_type}' split.")
            continue

        try:
            coco_dataset_instance = COCODataset(
                annotation_file=str(current_coco_json_path),
                image_root_dir=str(current_images_path),
                word_pickles_split_dir=str(current_word_pickles_path)
            )
            num_images_in_split = len(coco_dataset_instance)
            print(f"Found {num_images_in_split} processable images in '{dataset_type}' after initial filtering.")
            
            if num_images_in_split == 0:
                print(f"No images to process for '{dataset_type}'. Skipping this split.")
                continue
                
        except Exception as e:
            print(f"Error during COCODataset initialization for '{dataset_type}': {e}. Skipping this split.")
            continue
        
        # Process this split in chunks
        success = process_split_in_chunks(coco_dataset_instance, dataset_type, final_output_root)
        
        if not success:
            print(f"Failed to process '{dataset_type}' split")
        else:
            print(f"Successfully completed processing '{dataset_type}' split")

    print("\n--- Dataset generation complete. ---")
    print(f"Output saved in: {final_output_root}")