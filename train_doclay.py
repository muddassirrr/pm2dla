import os
from datetime import datetime
from pathlib import Path
import datasets
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from pycocotools.coco import COCO
import pickle
import wandb
from utils import copy_weights
from torch.utils.data import WeightedRandomSampler
import json
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

load_dotenv(".env")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NUM_TRAIN_EPOCHS = 50  
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8 
SAVE_LIMIT = 2 
LOGGING_STEPS = 100
LEARNING_RATE = 1e-6

IGNORE_ID = -100
MAX_LENGTH = 512
PRETRAIN_MODEL_ID = "/home/muddassir/Desktop/LayoutAnalysis/server_data/models"
PRETRAIN_BACKBONE_ID = None
FREEZE_BACKBONE = False


RUN_NAME = "30epochs_DO"
EXPERIMENTS_DIR = Path("experiments")

TRAIN_DATA="/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/val"
VAL_DATA = "/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/val"
TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"
DATASET_NUM_PROC = 4
DEVICE = "cuda"
device = torch.device(DEVICE)

PUSH_TO_HUB = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "1") != "0" # Default to true
os.environ["WANDB_API_KEY"] = "1b8a2709dd809a976979fd5666514b05052bb8b5"
os.environ["WANDB_PROJECT"] = "Thesis"
NEW_MODEL_CARD_PREFIX = "saeed11b95/multi-modal-prompt-detection"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{RUN_NAME}"


CLASS_WEIGHTS = {
    'Caption': 0.0849,
    'Footnote': 0.3881, 
    'Formula': 0.0767,
    'ListItem': 0.0400,
    'PageFooter': 0.0365,
    'PageHeader': 0.0339,
    'Picture': 0.0610,
    'SectionHeader': 0.0137,
    'Table': 0.0540,
    'Text': 0.0538,
    'Title': 0.2574,  
}

ID2LABEL = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "ListItem", 5: "PageFooter",
    6: "PageHeader", 7: "Picture", 8: "SectionHeader", 9: "Table", 10: "Text", 11: "Title",
}

PROMT_TO_CLASS = {
    "<CAP>": "Caption", "<FN>": "Footnote", "<FRM>": "Formula", "<LST>": "ListItem",
    "<PGF>": "PageFooter", "<PGH>": "PageHeader", "<PIC>": "Picture", "<SHD>": "SectionHeader",
    "<TAB>": "Table", "<TXT>": "Text", "<TTL>": "Title",
}

CLASS_TO_PROMPT = {v: k for k, v in PROMT_TO_CLASS.items()}

def get_augmentation_pipeline():
    """
    Define augmentation pipeline using albumentations.
    Only applied to images, not to bounding boxes or other data.
    """
    return A.Compose([
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(variance_limit=(10.0, 50.0), p=0.3),  # Fixed: var_limit → variance_limit
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),  # Fixed: single values → tuples
        ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=2, p=0.3),
    ], p=0.8)

def sort_class_instances_reading_order(class_boxes):
    """
    Sort boxes of a specific class in reading order (top to bottom, left to right).
    Takes only the boxes of the target class, not all boxes.
    """
    if not class_boxes:
        return class_boxes
    
    # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
    # Using the top-left corner of each box (xmin, ymin)
    sorted_boxes = sorted(class_boxes, key=lambda box: (box[1], box[0]))  # (ymin, xmin)
    
    return sorted_boxes

def quant_bbox_for_class_new_format(width, height, boxes, category_ids, target_class_name):
    """
    Generate bbox strings for a specific target class with new format.
    Each instance gets its own class label: ClassName<coords>ClassName<coords>...
    """
    bins_w, bins_h = [1000, 1000]
    size_per_bin_w = width / bins_w
    size_per_bin_h = height / bins_h
    
    # First, collect all boxes that belong to the target class
    target_class_boxes = []
    for bbox, cat_id in zip(boxes, category_ids):
        if ID2LABEL[cat_id] == target_class_name:
            target_class_boxes.append(bbox)
    
    # Sort only the instances of the target class in reading order
    sorted_class_boxes = sort_class_instances_reading_order(target_class_boxes)
    
    bbox_str = ""
    
    for bbox in sorted_class_boxes:
        bbox = bbox.copy()
        xmin, ymin, xmax, ymax = torch.tensor(bbox).split(1, dim=-1)
        quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
        quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        # New format: ClassName<coords> for each instance
        bbox_str += f"{target_class_name}<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>"
    
    return bbox_str

def create_training_targets_new_format(width, height, boxes, category_ids):
    """
    Create proper training targets for each class present in the image.
    Uses new format where each instance has its own class label.
    """
    present_classes = {ID2LABEL[cat_id] for cat_id in category_ids if cat_id in ID2LABEL}
    
    class_strings = []
    for class_name in present_classes:
        bbox_str = quant_bbox_for_class_new_format(width, height, boxes, category_ids, class_name)
        # Only add if the class actually has bounding boxes
        if bbox_str and class_name in bbox_str:
            class_strings.append(bbox_str)
    
    return class_strings

def paddbbox_seq(bboxs):
    padd_value = [[0, 0, 0, 0]]
    max_len_seq = max((len(boxes) for boxes in bboxs), default=0)
    padded_bboxs = [boxes + padd_value * (max_len_seq - len(boxes)) for boxes in bboxs]
    return padded_bboxs

def padd_input_ids(input_ids):
    padd_value = [0]
    max_len_seq = max((len(inp_ids) for inp_ids in input_ids), default=0)
    padded_ids = [inp_ids + padd_value * (max_len_seq - len(inp_ids)) for inp_ids in input_ids]
    return padded_ids

def sample_balanced_class_new_format(class_strings, class_weights):
    """
    Sample based on weights from new format class strings.
    """
    if not class_strings:
        return None
        
    class_names = []
    for class_str in class_strings:
        # Extract the first class name from the string
        # New format: ClassName<coords>ClassName<coords>...
        if class_str:
            # Find the first class name (before the first <loc_)
            first_class = class_str.split("<loc_")[0]
            # Remove any trailing characters and get the class name
            for class_name in ID2LABEL.values():
                if first_class.startswith(class_name):
                    class_names.append(class_name)
                    break
    
    if not class_names:
        return random.choice(class_strings) # Fallback
        
    weights = [class_weights.get(clss, 0.01) for clss in class_names]
    total = sum(weights)
    
    if total == 0:
        return random.choice(class_strings)
        
    probabilities = [w / total for w in weights]
    selected_idx = random.choices(range(len(class_strings)), weights=probabilities, k=1)[0]
    
    return class_strings[selected_idx]

def apply_augmentation(image, augmentation_pipeline, is_training=True):
    """
    Apply augmentation to image only during training.
    """
    if not is_training:
        return image
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Apply augmentation
    augmented = augmentation_pipeline(image=img_array)
    
    # Convert back to PIL image
    return Image.fromarray(augmented['image'])

def extract_class_name_from_string(class_str):
    """
    Extract class name from the new format string.
    """
    if not class_str:
        return None
    
    # Find the class name before the first <loc_
    first_part = class_str.split("<loc_")[0]
    
    # Check which class name it starts with
    for class_name in ID2LABEL.values():
        if first_part.startswith(class_name):
            return class_name
    
    return None

def collate_fn(batch):
    """
    Updated collate function with new format and augmentation.
    Generates labels on the fly from boxes and category_ids.
    """
    # Initialize augmentation pipeline
    augmentation_pipeline = get_augmentation_pipeline()
    
    processed_batch = []
    for example in batch:
        # Generate training targets on the fly
        if 'boxes' in example and 'category_ids' in example and 'image' in example:
            width, height = example['image'].width, example['image'].height
            
            # Generate class strings using the new format
            class_strings = create_training_targets_new_format(
                width, height, example['boxes'], example['category_ids']
            )
            
            # Apply augmentation to image only
            augmented_image = apply_augmentation(example['image'], augmentation_pipeline, is_training=True)
            
            processed_batch.append({
                **example, 
                'class_strs': class_strings,
                'image': augmented_image
            })
        else:
            # Skip examples without required data
            continue
    
    if not processed_batch:
        return None # Return None if batch is empty after filtering
        
    label_texts, prompt_texts = [], []
    valid_examples = []

    for example in processed_batch:
        selected_class_str = sample_balanced_class_new_format(example['class_strs'], CLASS_WEIGHTS)
        if selected_class_str:
            class_name = extract_class_name_from_string(selected_class_str)
            if class_name:
                prompt = CLASS_TO_PROMPT.get(class_name, "<UNK>")
                
                label_texts.append(selected_class_str)
                prompt_texts.append(prompt)
                valid_examples.append(example)

    if not label_texts:
        return None # Return None if no valid labels could be generated

    images = [ex["image"] for ex in valid_examples]
    input_ids = padd_input_ids([ex["input_ids"] for ex in valid_examples])
    subword_bboxs = paddbbox_seq([ex["subword_bboxs"] for ex in valid_examples])

    grid_data = [
        {'input_ids': torch.tensor(inp_id), 'bbox': torch.tensor(rescale_bboxs(torch.tensor(bbox)))}
        for inp_id, bbox in zip(input_ids, subword_bboxs)
    ]

    inputs = processor(
        images=images, text=prompt_texts, return_tensors="pt",
        padding="longest", max_length=MAX_LENGTH,
    )

    labels = processor.tokenizer(
        label_texts, return_tensors="pt", padding="longest",
        max_length=MAX_LENGTH, return_token_type_ids=False,
    )["input_ids"]

    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
    print(f"Generated labels: {labels}")
    
    return {**inputs, "labels": labels, "grid_data": grid_data}


def rescale_bboxs(bboxs):
    scale = 768 / 1025
    return [[x * scale for x in box] for box in bboxs]

def validate_dataset_structure(dataset, split_name):
    """
    Updated validation function - no longer checks for class_strs.
    """
    print(f"\n=== Checking {split_name} dataset structure ===")
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return False
    
    # Check first few examples for structure
    sample_size = min(10, len(dataset))
    common_keys = None
    
    for i in range(sample_size):
        try:
            example = dataset[i]
            if common_keys is None:
                common_keys = set(example.keys())
            else:
                common_keys &= set(example.keys())
        except Exception as e:
            print(f"Error accessing sample {i}: {e}")
            return False
    
    print(f"Common keys across samples: {sorted(common_keys)}")
    
    # Check for expected keys (removed class_strs from requirements)
    expected_keys = ["image", "boxes", "category_ids", "input_ids", "subword_bboxs"]
    missing_keys = [key for key in expected_keys if key not in common_keys]
    
    if missing_keys:
        print(f"⚠️  Missing expected keys: {missing_keys}")
    else:
        print("✅ All expected keys present")
    
    return len(missing_keys) == 0

from tqdm import tqdm
import datasets

def validate_and_filter_dataset(dataset, split_name, required_keys=None):
    """
    Updated validation and filtering function.
    No longer requires class_strs as they're generated on the fly.
    """
    if required_keys is None:
        required_keys = ["image", "boxes", "category_ids", "input_ids", "subword_bboxs"]
    
    print(f"\n=== Validating and filtering {split_name} dataset ===")
    print(f"Total samples before filtering: {len(dataset)}")
    
    valid_indices = []
    invalid_reasons = {
        "missing_keys": 0,
        "empty_boxes": 0,
        "invalid_image": 0,
        "invalid_ocr_data": 0,
        "mismatched_lengths": 0,
        "other_errors": 0
    }
    
    class_counts = {}
    
    # Validate each example with progress bar
    for i, example in enumerate(tqdm(dataset, desc=f"Validating {split_name}")):
        try:
            # Check for required keys
            missing_keys = [key for key in required_keys if key not in example]
            if missing_keys:
                invalid_reasons["missing_keys"] += 1
                continue
            
            # Check if image is valid
            if example["image"] is None:
                invalid_reasons["invalid_image"] += 1
                continue
            
            # Check if boxes and category_ids are valid and not empty
            if not example.get("boxes") or not example.get("category_ids"):
                invalid_reasons["empty_boxes"] += 1
                continue
            
            # Check if boxes and category_ids have matching lengths
            if len(example["boxes"]) != len(example["category_ids"]):
                invalid_reasons["mismatched_lengths"] += 1
                continue
            
            # Check OCR data validity
            if not example.get("input_ids") or not example.get("subword_bboxs"):
                invalid_reasons["invalid_ocr_data"] += 1
                continue
            
            # Check if input_ids and subword_bboxs have matching lengths
            if len(example["input_ids"]) != len(example["subword_bboxs"]):
                invalid_reasons["invalid_ocr_data"] += 1
                continue
            
            # Count classes for statistics
            for cat_id in example["category_ids"]:
                if cat_id in ID2LABEL:
                    class_name = ID2LABEL[cat_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            invalid_reasons["other_errors"] += 1
            continue
    
    # Create filtered dataset
    if valid_indices:
        filtered_dataset = dataset.select(valid_indices)
    else:
        print("❌ No valid samples found!")
        return None
    
    # Report statistics
    print(f"Valid samples after filtering: {len(filtered_dataset)}")
    print(f"Filtered out: {len(dataset) - len(filtered_dataset)} samples")
    
    if invalid_reasons:
        print("\nInvalid sample breakdown:")
        for reason, count in invalid_reasons.items():
            if count > 0:
                percentage = (count / len(dataset)) * 100
                print(f"  {reason}: {count} ({percentage:.2f}%)")
    
    if class_counts:
        print(f"\nClass distribution in valid samples:")
        total_class_instances = sum(class_counts.values())
        for cls, count in sorted(class_counts.items()):
            percentage = (count / total_class_instances) * 100
            print(f"  {cls}: {count} ({percentage:.2f}%)")
    
    return filtered_dataset

# --- Main Training Execution ---

if __name__ == "__main__":
    # Create experiments directory if it doesn't exist
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    # Login to W&B if enabled
    if LOG_TO_WANDB:
        try:
            wandb.login()
            print("✅ Successfully logged into W&B.")
        except Exception as e:
            print(f"Could not log in to W&B. Please check your API key. Error: {e}")
            LOG_TO_WANDB = False # Disable W&B if login fails

    train_dataset = datasets.load_from_disk(TRAIN_DATA, TRAIN_SPLIT_NAME)
    val_dataset = datasets.load_from_disk(VAL_DATA, VAL_SPLIT_NAME)

    # if not validate_dataset_structure(train_dataset, "train"):
    #     print("❌ Train dataset structure validation failed.")
    #     exit(1)
    
    # if not validate_dataset_structure(val_dataset, "validation"):
    #     print("❌ Validation dataset structure validation failed.")
    #     exit(1)
    
    # # Detailed validation and filtering
    # train_dataset = validate_and_filter_dataset(train_dataset, "train")
    # val_dataset = validate_and_filter_dataset(val_dataset, "validation")
    
    # if train_dataset is None or val_dataset is None:
    #     print("❌ Dataset validation failed.")
    #     exit(1)

    processor = AutoProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config.vision_config.model_type = "davit"
    new_model = AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True, config=config)

    # if PRETRAIN_BACKBONE_ID:
    #     # This part for loading a separate backbone is kept from the original script
    #     # Ensure BACKBONE_FILE_NAME is defined if you use this
    #     BACKBONE_FILE_NAME = "your_backbone_file.safetensors" 
    #     local_dir = EXPERIMENTS_DIR / "backbone"
    #     file_path = local_dir / BACKBONE_FILE_NAME
    #     hf_hub_download(PRETRAIN_BACKBONE_ID, BACKBONE_FILE_NAME, local_dir=local_dir)
    #     ckpt_state_dict = load_file(file_path)
    #     vision_state_dict = copy_weights(ckpt_state_dict, new_model.state_dict())
    #     new_model.load_state_dict(vision_state_dict)

    if FREEZE_BACKBONE:
        for param in new_model.vision_tower.parameters():
            param.requires_grad = False

    # Define Training Arguments
    args = TrainingArguments(
        output_dir=f"{EXPERIMENTS_DIR}/{RUN_NAME}",
        run_name=RUN_NAME,
        
        # Switched to epoch-based training
        num_train_epochs=NUM_TRAIN_EPOCHS,
        
        # Batch size and accumulation
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1,
        
        # Checkpointing and evaluation strategy
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_LIMIT,
        load_best_model_at_end=True,
        
        # Logging
        logging_steps=LOGGING_STEPS,
        logging_strategy="steps",
        report_to="wandb" if LOG_TO_WANDB else "none",
        
        # Other parameters
        fp16=True,
        learning_rate=LEARNING_RATE,
        dataloader_num_workers=6,
        label_names=["labels"],
        remove_unused_columns=False,
        
        # Hub settings
        push_to_hub=False, # Controlled manually at the end
        hub_model_id=NEW_MODEL_CARD,
    )

    trainer = Trainer(
        model=new_model,
        args=args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)],
    )

    print("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=None)
    print("🏁 Training finished.")

    # Save the best model, processor, and training logs
    best_ckpt_path = trainer.state.best_model_checkpoint
    final_model_path = EXPERIMENTS_DIR / f"{RUN_NAME}-final"
    print(f"Best checkpoint found at: {best_ckpt_path}")
    print(f"Saving final model to: {final_model_path}")
    
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    log_history = trainer.state.log_history
    log_file_path = EXPERIMENTS_DIR / f"{RUN_NAME}-training_log.json"
    with open(log_file_path, "w") as f:
        json.dump(log_history, f, indent=4)
    print(f"Training log saved to: {log_file_path}")

    if PUSH_TO_HUB:
        print(f"🚀 Pushing final model to Hugging Face Hub: {NEW_MODEL_CARD}")
        processor.push_to_hub(NEW_MODEL_CARD, private=True)
        trainer.push_to_hub(NEW_MODEL_CARD, private=True)

    if LOG_TO_WANDB:
        wandb.finish()

    print("✅ Training script completed successfully!")