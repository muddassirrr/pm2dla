import os
import random
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import datasets
import torch
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from transformers import (
   AutoConfig,
   AutoModelForCausalLM,
   AutoProcessor,
   EarlyStoppingCallback,
   Trainer,
   TrainingArguments,
)

load_dotenv(".env")

# Training Configuration
EVAL_BATCH_SIZE = 2
NUM_IMAGES_TO_TRAIN = 1400000
TRAIN_BATCH_SIZE = 3
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 4
SAVE_STEPS = 15000
LOGGING_STEPS = 100
IGNORE_ID = -100  # PyTorch ignore index when computing loss
MAX_LENGTH = 512
DEVICE = "cuda"
FREEZE_BACKBONE = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"

# Model and Dataset Configuration for PubLayNet
NEW_MODEL_CARD_PREFIX = "saeed11b95/florence2-publaynet-detection"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
PRETRAIN_MODEL_ID = "/home/docanalysis/florence2-training/models"
PROMPT = "<OD>"
PUSH_TO_HUB = False
RUN_NAME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Paths for PubLayNet
SAVE_DIRECTORY = Path("/opt/publaynet/runs")
TRAIN_DATA = '/opt/publaynet/Publaynet/dataset_florence2_publaynet_simple/train'
VAL_DATA = '/opt/publaynet/Publaynet/dataset_florence2_publaynet_simple/val'

# PubLayNet class and prompt mappings
ID2LABEL = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
PROMPT_TO_CLASS = {"<TXT>": "text", "<TTL>": "title", "<LST>": "list", "<TAB>": "table", "<FIG>": "figure"}
CLASS_TO_PROMPT = {v: k for k, v in PROMPT_TO_CLASS.items()}

# --- Pre-calculated Class Weights for Balanced Sampling ---
CLASS_WEIGHTS = {
   'text': 0.06202770713596694,
   'title': 0.08114653564675009,
   'list': 0.391189554460472,
   'table': 0.2400219045732182,
   'figure': 0.2256142981835927
}

# --- Balanced Sampling Helper Function ---

def sample_balanced_class(class_strings, class_weights):
   """Sample a class string from the available list based on pre-calculated weights."""
   if not class_strings:
       return None

   available_classes = [s.split('<')[0] for s in class_strings]
   weights_for_available = [class_weights.get(name, 0) for name in available_classes]
  
   if sum(weights_for_available) == 0:
       return random.choice(class_strings)

   # Use random.choices to sample one item using the weights
   return random.choices(class_strings, weights=weights_for_available, k=1)[0]


# --- Data Preprocessing and Collation ---

def paddbbox_seq(bboxs):
   """Pad bounding box sequences to same length."""
   padd_value = [[0, 0, 0, 0]]
   max_len_seq = max([len(boxes) for boxes in bboxs])
   padded_bboxs = []
   for boxes in bboxs:
       padded_bboxs.append(boxes + padd_value * (max_len_seq - len(boxes)))
   return padded_bboxs


def padd_input_ids(input_ids):
   """Pad input ID sequences to same length."""
   padd_value = [0]
   max_len_seq = max([len(inp_ids) for inp_ids in input_ids])
   padded_ids = []
   for inp_ids in input_ids:
       padded_ids.append(inp_ids + padd_value * (max_len_seq - len(inp_ids)))
   return padded_ids


def rescale_bboxs(bboxs):
   """Rescale bounding boxes."""
   scale = 768 / 1025
   return [[x * scale for x in box] for box in bboxs]


def create_collate_fn(processor, class_weights):
   """Creates the collate_fn with class_weights and processor in its scope."""

   def collate_fn(batch):
       """Data collation function for training with balanced sampling."""
       images = [Image.open(example["image_path"]).convert("RGB") for example in batch]
      
       label_texts = [sample_balanced_class(example["class_strs"], class_weights) for example in batch]
       prompt_texts = [CLASS_TO_PROMPT[label_text.split("<")[0]] for label_text in label_texts]

       # Prepare grid_data for OCR-aware features (FIX for the TypeError)
       ocr_input_ids = padd_input_ids([example["input_ids"] for example in batch])
       subword_bboxs = paddbbox_seq([example["subword_bboxs"] for example in batch])
       grid_data = [{'input_ids': torch.tensor(input_id), 'bbox': torch.tensor(rescale_bboxs(subword_bbox))}
                    for input_id, subword_bbox in zip(ocr_input_ids, subword_bboxs)]

       # Process main prompt and image
       inputs = processor(
           images=images, text=prompt_texts, return_tensors="pt",
           padding="longest", max_length=MAX_LENGTH
       )
      
       # Process labels
       labels = processor.tokenizer(
           label_texts, return_tensors="pt", padding="longest",
           max_length=MAX_LENGTH, return_token_type_ids=False
       )["input_ids"]

       labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
      
       # Return all required inputs for the model
       return {**inputs, "labels": labels, "grid_data": grid_data}

   return collate_fn


if __name__ == "__main__":
   SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

   print("Loading datasets...")
   train_dataset = datasets.load_from_disk(TRAIN_DATA)
   val_dataset = datasets.load_from_disk(VAL_DATA)
   print(f"Train dataset size: {len(train_dataset)}")
   print(f"Val dataset size: {len(val_dataset)}")

   print("Loading processor and model...")
   processor = AutoProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
   config = AutoConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
   config.vision_config.model_type = "davit"
   model = AutoModelForCausalLM.from_pretrained(
       PRETRAIN_MODEL_ID, trust_remote_code=True, config=config
   )
  
   print("Using pre-calculated class weights for sampling.")
   print(f"Class Weights: {CLASS_WEIGHTS}")

   # Create the data collator with the pre-calculated weights
   data_collator = create_collate_fn(processor, CLASS_WEIGHTS)

   if FREEZE_BACKBONE:
       for param in model.vision_tower.parameters():
           param.requires_grad = False

   args = TrainingArguments(
       dataloader_num_workers=6,
       eval_steps=SAVE_STEPS,
       eval_strategy="steps",
       fp16=True,
       label_names=["labels"],
       learning_rate=1e-6,
       load_best_model_at_end=True,
       logging_steps=LOGGING_STEPS,
       logging_strategy="steps",
       max_steps=TRAIN_STEPS,
       output_dir=f"{SAVE_DIRECTORY}/{RUN_NAME}",
       per_device_eval_batch_size=EVAL_BATCH_SIZE,
       per_device_train_batch_size=TRAIN_BATCH_SIZE,
       push_to_hub=PUSH_TO_HUB,
       remove_unused_columns=False, # Important: keep original columns for grid_data
       report_to="none" if not LOG_TO_WANDB else "wandb",
       run_name=RUN_NAME,
       save_steps=SAVE_STEPS,
       save_strategy="steps",
       save_total_limit=SAVE_LIMIT,
       hub_model_id=NEW_MODEL_CARD,
   )

   trainer = Trainer(
       args=args,
       callbacks=[EarlyStoppingCallback(early_stopping_patience=310, early_stopping_threshold=0.0001)],
       data_collator=data_collator,
       model=model,
       # The 'tokenizer' argument is deprecated, but we pass processor here
       # which has the tokenizer. This is the standard practice.
       tokenizer=processor,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
   )

   print("Starting training...")
   trainer.train(resume_from_checkpoint=None)

   best_ckpt_path = trainer.state.best_model_checkpoint
   final_model_path = Path("/opt/publaynet/florence2_publaynet_final")
   print(f"Best checkpoint: {best_ckpt_path}")
   print(f"Saving final model to: {final_model_path}")

   trainer.save_model(str(final_model_path))
   processor.save_pretrained(str(final_model_path))

   # Save training logs
   log_history = trainer.state.log_history
   log_file_path = final_model_path / "training_log.json"
   print(f"Saving training log to: {log_file_path}")
   with open(log_file_path, "w") as f:
       json.dump(log_history, f, indent=4)

   if PUSH_TO_HUB:
       processor.push_to_hub(NEW_MODEL_CARD, private=True)
       trainer.push_to_hub(NEW_MODEL_CARD, private=True)

   if LOG_TO_WANDB and 'wandb' in __import__('sys').modules:
       import wandb
       wandb.finish()
      
   print("✅ Training completed successfully!")

