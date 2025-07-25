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

# --- Configuration ---
load_dotenv(".env")

# Set specific GPUs to use, e.g., "0,1" for the first two GPUs
# The trainer will automatically use DataParallel or DistributedDataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

NUM_TRAIN_EPOCHS = 50  
TRAIN_BATCH_SIZE = 5
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 3 # Configure gradient accumulation
SAVE_LIMIT = 2  # Set the total number of checkpoints to save
LOGGING_STEPS = 100
LEARNING_RATE = 1e-6

# Model and Tokenizer settings
IGNORE_ID = -100
MAX_LENGTH = 512
PRETRAIN_MODEL_ID = "/data2/Saeed_Data/code_modularized/models"
PRETRAIN_BACKBONE_ID = None
FREEZE_BACKBONE = False

# I/O and Logging
RUN_NAME = "50epoch"
EXPERIMENTS_DIR = Path("experiments") # All models will be saved here
TRAIN_DATA = '/data2/Saeed_Data/code_modularized/dataset_florence2_td/train'
VAL_DATA = '/data2/Saeed_Data/code_modularized/dataset_florence2_td/val'
TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"
DATASET_NUM_PROC = 4
DEVICE = "cuda"
device = torch.device(DEVICE)

# Hugging Face Hub and W&B settings
PUSH_TO_HUB = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "1") != "0" # Default to true
# Add your W&B API key here if not set as an environment variable
os.environ["WANDB_API_KEY"] = "1b8a2709dd809a976979fd5666514b05052bb8b5"
os.environ["WANDB_PROJECT"] = "Thesis"
NEW_MODEL_CARD_PREFIX = "saeed11b95/multi-modal-prompt-detection"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{RUN_NAME}"


CLASS_WEIGHTS = {
    'Caption': 0.0849,
    'Footnote': 0.4881,  # Increased from 0.2881
    'Formula': 0.0767,
    'ListItem': 0.0100,
    'PageFooter': 0.0265,
    'PageHeader': 0.0339,
    'Picture': 0.0410,
    'SectionHeader': 0.0137,
    'Table': 0.0540,
    'Text': 0.0038,
    'Title': 0.2674,  # Reduced from 0.3674 to balance
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


# --- Data Processing Functions ---

def quant_bbox_for_class(width, height, boxes, category_ids, target_class_name):
    """
    Generate bbox strings for a specific target class only.
    """
    bins_w, bins_h = [1000, 1000]
    size_per_bin_w = width / bins_w
    size_per_bin_h = height / bins_h
    
    bbox_str = target_class_name
    
    for bbox, cat_id in zip(boxes, category_ids):
        if ID2LABEL[cat_id] != target_class_name:
            continue
            
        bbox = bbox.copy()
        xmin, ymin, xmax, ymax = torch.tensor(bbox).split(1, dim=-1)
        quantized_xmin = (xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymin = (ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
        quantized_xmax = (xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
        quantized_ymax = (ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        bbox_str += f"<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>"
    
    return bbox_str

def create_training_targets(width, height, boxes, category_ids):
    """
    Create proper training targets for each class present in the image.
    """
    present_classes = {ID2LABEL[cat_id] for cat_id in category_ids if cat_id in ID2LABEL}
    
    class_strings = []
    for class_name in present_classes:
        bbox_str = quant_bbox_for_class(width, height, boxes, category_ids, class_name)
        # Only add if the class actually has bounding boxes
        if "<loc_" in bbox_str:
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

def sample_balanced_class(class_strings, class_weights):
    """
    Properly extract class names and sample based on weights.
    """
    if not class_strings:
        return None
        
    class_names = []
    for class_str in class_strings:
        class_name = class_str.split("<loc_")[0]
        class_names.append(class_name)
    
    if not class_names:
        return random.choice(class_strings) # Fallback
        
    weights = [class_weights.get(cls, 0.01) for cls in class_names]
    total = sum(weights)
    
    if total == 0:
        return random.choice(class_strings)
        
    probabilities = [w / total for w in weights]
    selected_idx = random.choices(range(len(class_strings)), weights=probabilities, k=1)[0]
    
    return class_strings[selected_idx]

def collate_fn(batch):
    """
    Updated collate function with proper data handling.
    """
    processed_batch = []
    for example in batch:
        if 'boxes' in example and 'category_ids' in example:
            width, height = example['image'].width, example['image'].height
            class_strings = create_training_targets(width, height, example['boxes'], example['category_ids'])
        else:
            class_strings = example.get('class_strs', [])
        
        if class_strings:
            processed_batch.append({**example, 'class_strs': class_strings})
    
    if not processed_batch:
        return None # Return None if a batch is empty after filtering
        
    label_texts, prompt_texts = [], []
    valid_examples = []

    for example in processed_batch:
        selected_class_str = sample_balanced_class(example['class_strs'], CLASS_WEIGHTS)
        if selected_class_str:
            class_name = selected_class_str.split("<loc_")[0]
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
    
    return {**inputs, "labels": labels, "grid_data": grid_data}


def rescale_bboxs(bboxs):
    scale = 768 / 1025
    return [[x * scale for x in box] for box in bboxs]

def validate_dataset(dataset, split_name):
    """
    Validate dataset structure and report statistics.
    """
    print(f"\n=== Validating {split_name} dataset ===")
    print(f"Total samples: {len(dataset)}")
    
    class_counts = {cls: 0 for cls in ID2LABEL.values()}
    valid_samples = 0
    
    for i, example in enumerate(dataset):
        try:
            if 'category_ids' in example and 'boxes' in example:
                valid_samples += 1
                for cat_id in example['category_ids']:
                    if cat_id in ID2LABEL:
                        class_counts[ID2LABEL[cat_id]] += 1
        except Exception as e:
            print(f"Error in sample {i}: {e}")
    
    print(f"Valid samples: {valid_samples}")
    print("Class distribution:")
    total_annotations = sum(class_counts.values())
    if total_annotations > 0:
        for cls, count in sorted(class_counts.items()):
            percentage = (count / total_annotations) * 100
            print(f"  {cls}: {count} ({percentage:.2f}%)")
    
    return valid_samples > 0

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
    
    # Validate datasets before proceeding
    # if not validate_dataset(train_dataset, "train") or not validate_dataset(val_dataset, "validation"):
    #     print("❌ Dataset validation failed. Please check your data format.")
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # Checkpointing and evaluation strategy
        evaluation_strategy="epoch",
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