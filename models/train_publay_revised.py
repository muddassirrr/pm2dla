import os
import random
import json
from datetime import datetime
from pathlib import Path

import datasets
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from modeling_pmdla import PMDLAForConditionalGeneration
from configuration_pmdla import PMDLAConfig
from processing_pmdla import PMDLAProcessor

load_dotenv(".env")

EVAL_BATCH_SIZE = 2
NUM_IMAGES_TO_TRAIN = 1_400_000
TRAIN_BATCH_SIZE = 3
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 4
SAVE_STEPS = 15_000
LOGGING_STEPS = 100
IGNORE_ID = -100
MAX_LENGTH = 512
DEVICE = "cuda"
FREEZE_BACKBONE = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"

NEW_MODEL_CARD_PREFIX = "saeed11b95/pmdla-publaynet-detection"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
# PRETRAIN_MODEL_ID = "/home/docanalysis/florence2-training/models"
PRETRAIN_MODEL_ID = "/home/muddassir/Desktop/LayoutAnalysis/code_modularized/models"
PUSH_TO_HUB = False
RUN_NAME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

SAVE_DIRECTORY = Path("/home/muddassir/Desktop/LayoutAnalysis/experiments/runs")
TRAIN_DATA = '/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/train'
VAL_DATA = '/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/val'
ID2LABEL = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}
CLASS_TO_PROMPT = {"text": "<TXT>", "title": "<TTL>", "list": "<LST>", "table": "<TAB>", "figure": "<FIG>"}

CLASS_WEIGHTS = {
   'text': 0.062, 'title': 0.081, 'list': 0.391, 'table': 0.240, 'figure': 0.225
}

def sample_balanced_class(class_strings, class_weights):
   """Sample a class string from the available list based on pre-calculated weights."""
   if not class_strings:
       return None
   available_classes = [s.split('<')[0] for s in class_strings]
   weights_for_available = [class_weights.get(name, 0) for name in available_classes]
   if sum(weights_for_available) == 0:
       return random.choice(class_strings)
   return random.choices(class_strings, weights=weights_for_available, k=1)[0]

def paddbbox_seq(bboxs):
   """Pad bounding box sequences to the same length."""
   max_len = max(len(boxes) for boxes in bboxs)
   return [boxes + [[0, 0, 0, 0]] * (max_len - len(boxes)) for boxes in bboxs]

def padd_input_ids(input_ids):
   """Pad input ID sequences to the same length."""
   max_len = max(len(ids) for ids in input_ids)
   return [ids + [0] * (max_len - len(ids)) for ids in input_ids]

def rescale_bboxs(bboxs, scale=768 / 1025):
   """Rescale bounding boxes."""
   return [[coord * scale for coord in box] for box in bboxs]

def create_collate_fn(processor, class_weights):
   """Creates the collate_fn with class_weights and processor in its scope."""
   def collate_fn(batch):
       images = [Image.open(example["image_path"]).convert("RGB") for example in batch]
       
       label_texts = [sample_balanced_class(example["class_strs"], class_weights) for example in batch]
       prompt_texts = [CLASS_TO_PROMPT[label.split("<")[0]] for label in label_texts]
       
       # Note: The collate_fn always prepares grid_data.
       # The model will internally decide whether to use it based on its configuration.
       # This makes the training script robust and reusable for different model architectures.
       ocr_input_ids = padd_input_ids([example["input_ids"] for example in batch])
       subword_bboxs = paddbbox_seq([example["subword_bboxs"] for example in batch])
       grid_data = [{'input_ids': torch.tensor(ids), 'bbox': torch.tensor(rescale_bboxs(bbox))}
                    for ids, bbox in zip(ocr_input_ids, subword_bboxs)]

       inputs = processor(
           images=images, text=prompt_texts, return_tensors="pt",
           padding="longest", max_length=MAX_LENGTH
       )
       
       labels = processor.tokenizer(
           label_texts, return_tensors="pt", padding="longest",
           max_length=MAX_LENGTH, return_token_type_ids=False
       )["input_ids"]
       labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
       
       # The model's forward pass is flexible. It will accept 'grid_data' but only
       # use it if the 'grid' backbone is enabled in the model's config.
       return {**inputs, "labels": labels, "grid_data": grid_data}
   return collate_fn

if __name__ == "__main__":
   SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

   print("Loading datasets...")
   train_dataset = datasets.load_from_disk(TRAIN_DATA)
   val_dataset = datasets.load_from_disk(VAL_DATA)
   print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

   print("Loading processor and model...")
   processor = PMDLAProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
   
   # --- Model loading is now config-driven ---
   # You can now modify the config before loading the model to change its architecture.
   # For example:
   # config = PMDLAConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
   # config.enabled_backbones = ["vision"]  # Train with only the vision backbone
   # config.fusion_strategy = "concatenate" # This would require a compatible checkpoint
   # model = PMDLAForConditionalGeneration.from_pretrained(
   #     PRETRAIN_MODEL_ID, trust_remote_code=True, config=config
   # )
   config = PMDLAConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
   model = PMDLAForConditionalGeneration.from_pretrained(
       PRETRAIN_MODEL_ID, trust_remote_code=True, config=config
   )
  
   data_collator = create_collate_fn(processor, CLASS_WEIGHTS)

   if FREEZE_BACKBONE:
       # This part remains the same, but you might want to conditionally freeze
       # the grid_tower as well if it exists.
       if model.vision_tower:
           for param in model.vision_tower.parameters():
               param.requires_grad = False
       if model.grid_tower:
           for param in model.grid_tower.parameters():
               param.requires_grad = False


   training_args = TrainingArguments(
       dataloader_num_workers=6,
       eval_steps=SAVE_STEPS,
       evaluation_strategy="steps",
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
       remove_unused_columns=False,
       report_to="wandb" if LOG_TO_WANDB else "none",
       run_name=RUN_NAME,
       save_steps=SAVE_STEPS,
       save_strategy="steps",
       save_total_limit=SAVE_LIMIT,
       hub_model_id=NEW_MODEL_CARD,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       callbacks=[EarlyStoppingCallback(early_stopping_patience=310, early_stopping_threshold=0.0001)],
       data_collator=data_collator,
       tokenizer=processor,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
   )

   print("Starting training...")
   trainer.train()

   best_ckpt_path = trainer.state.best_model_checkpoint
   final_model_path = Path("/opt/publaynet/pmdla_publaynet_final")
   print(f"Best checkpoint: {best_ckpt_path}")
   print(f"Saving final model to: {final_model_path}")

   trainer.save_model(str(final_model_path))
   processor.save_pretrained(str(final_model_path))

   log_history_path = final_model_path / "training_log.json"
   print(f"Saving training log to: {log_history_path}")
   with open(log_history_path, "w") as f:
       json.dump(trainer.state.log_history, f, indent=4)

   if PUSH_TO_HUB:
       print(f"Pushing model and processor to the Hub: {NEW_MODEL_CARD}")
       processor.push_to_hub(NEW_MODEL_CARD, private=True)
       trainer.push_to_hub(NEW_MODEL_CARD, private=True)

   if LOG_TO_WANDB and 'wandb' in __import__('sys').modules:
       import wandb
       wandb.finish()
      
   print("✅ Training completed successfully!")
