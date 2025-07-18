import io
import shutil
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

# base_dir = Path("/media/saeed/cb770179-4e9f-49f3-b105-5be4357ac72e/Doclaynet/FullData")
base_dir = Path("/home/muddassir/Desktop/LayoutAnalysis/datasets/DocLayNet_core")

coco_dir = base_dir / "COCO"
# word_pickle_path = base_dir / "word_pickles"
word_pickle_path="/home/muddassir/Desktop/LayoutAnalysis/datasets/Doclaynet_word/VGT_DocLayNet_grid_pkl"
id2label = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",
    4: "ListItem",
    5: "PageFooter",
    6: "PageHeader",
    7: "Picture",
    8: "SectionHeader",
    9: "Table",
    10: "Text",
    11: "Title",
}
features = Features(
    {
        "image": datasets.Image(),
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
NUM_BATCHES = 8
print("NUM_BATCHES", NUM_BATCHES)



def quant_bbox(width, height, boxes, category_ids):
    bins_w, bins_h = [1000, 1000]  # Quantization bins.
    size_per_bin_w = width / bins_w
    size_per_bin_h = height / bins_h

    bbox_str_dict = {cat_id: id2label[cat_id] for cat_id in category_ids}
    for bbox, cat_id in zip(boxes, category_ids):
        bbox = bbox.copy()

        xmin, ymin, xmax, ymax = torch.tensor(bbox).split(1, dim=-1)
        quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
        quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        bbox_str_dict[cat_id] += (
            f"<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>"
        )

    full_bbox_str = ""
    class_strs = []
    for bbox_str in bbox_str_dict.values():
        full_bbox_str += bbox_str
        class_strs.append(bbox_str)
    return full_bbox_str, class_strs


class COCODataset(torch_dataset):
    def __init__(self, annotation_file: str, image_dir: str, word_pickles_dir, image_size=(1025, 1025)):
        print(coco_dir)
        self.coco = COCO(annotation_file)
        self.image_dir = Path(image_dir)
        self.word_pickles_dir = Path(word_pickles_dir)
        image_ids = self.coco.getImgIds()
        ids_to_remove = self.filter_img_ids(image_ids)
        self.image_ids = [img_id for img_id in image_ids if img_id not in ids_to_remove]

    def filter_img_ids (self, img_ids):
        ids_to_remove = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = self.image_dir / img_info["file_name"]
            if not img_path.exists():
                print(f"Image {img_path} does not exist")
                ids_to_remove.append(img_id)
        return ids_to_remove
        
    def __len__(self):
        return len(self.image_ids)

    def _load_word_pickle(self, image_info):
        file_name = image_info["file_name"][0:-4] + ".pdf.pkl"
        word_pickle_path = self.word_pickles_dir / file_name
        with open(word_pickle_path, "rb") as f:
            word_info = pickle.load(f)
        return word_info

    def __getitem__(self, idx_list):
        is_int = isinstance(idx_list, int)
        if is_int:
            idx_list = [idx_list]

        outputs = []
        for idx in idx_list:
            image_id = self.image_ids[idx]
            image_info = self.coco.imgs[image_id]
            image_path = self.image_dir / image_info["file_name"]
            image = Image.open(image_path)
            word_info = self._load_word_pickle(image_info)
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
            boxes = [ann["bbox"] for ann in annotations]
            category_ids = [ann["category_id"] for ann in annotations]
            boxes = [
                [box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes
            ]
            width, height = image.size
            bbox_str, class_strs = quant_bbox(width, height, boxes, category_ids)

            output = {
                "image": image,
                "category_ids": category_ids,
                "image_id": image_id,
                "boxes": boxes,
                "width": width,
                "height": height,
                "bbox_str": bbox_str,
                "class_strs": class_strs,
                "ocr_words": word_info["texts"],
                "ocr_boxes": word_info["bbox_texts_list"],
                "input_ids": word_info["input_ids"],
                "subword_bboxs": word_info["bbox_subword_list"],
            }
            assert sorted(output.keys()) == sorted(features.keys())

            outputs.append(output)

        if is_int:
            return outputs[0]
        return outputs


def generator_fn(shards):
    print("Shards", shards)
    dataset_type = shards[0][1]
    dataset = COCODataset(coco_dir / f"{dataset_type}.json", img_dir, word_pickle_path)
    batch_num = shards[0][0]
    batch_size = len(dataset) // NUM_BATCHES
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size
    for i in range(start_idx, end_idx):
        yield dataset[i]


if __name__ == "__main__":
    combined_dataset = DatasetDict()
    for dataset_type in ["train", "val", "test"]:
        img_dir = base_dir / f"{dataset_type}/images"
        dataset = COCODataset(coco_dir / f"{dataset_type}.json", img_dir, word_pickle_path)
        print(f"Number of images in {dataset_type}: {len(dataset)}")

        dataset = Dataset.from_generator(
            generator_fn,
            features=features,
            gen_kwargs={"shards": [(i, dataset_type) for i in range(NUM_BATCHES)]},
            num_proc=NUM_BATCHES,
        )

        shutil.rmtree(f"./dataset_florence2_td/{dataset_type}", ignore_errors=True)
        dataset.save_to_disk(
            f"./dataset_florence2_td/{dataset_type}", num_proc=NUM_BATCHES
        )

        # combined_dataset[dataset_type] = dataset

        # dataset.push_to_hub(
        #     "katphlab/doclaynet-table",
        #     config_name=dataset_type,
        #     split=dataset_type,
        # )
