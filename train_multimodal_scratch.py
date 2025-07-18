
import os
from datetime import datetime
from pathlib import Path
import math
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
from torchinfo import summary

load_dotenv(".env")
EVAL_BATCH_SIZE = 2
TRAIN_BATCH_SIZE = 3 # Per device batch size
# --- Epoch Settings ---
NUM_EPOCHS_TO_TRAIN = 100
SAVE_EVERY_N_EPOCHS = 10
EVAL_EVERY_N_EPOCHS = 10 # Align evaluation with saving
# ---------
# TRAIN_STEPS is now determined by epochs and dataset size, remove or comment out manual setting
# TRAIN_STEPS = 500000
SAVE_LIMIT = 4 # Keeps the latest 4 checkpoints based on save frequency
LOGGING_STEPS = 100 # Log metrics every 100 steps (can keep this step-based)
IGNORE_ID = -100
MAX_LENGTH = 512
DEVICE = "cuda"
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
PRETRAIN_MODEL_ID = "/home/docanalysis/florence2-training/models"
TRAIN_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/train'
VAL_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/val'
NEW_MODEL_CARD_PREFIX = "saeed11b95/multi-modal-prompt-detection"

PRETRAIN_BACKBONE_ID = None
FREEZE_BACKBONE = False

SCRATCH_TRAIN_DIR_NAME = "trained_scratch"
SAVE_DIRECTORY = Path("./runs")
SCRATCH_OUTPUT_PATH = SAVE_DIRECTORY / SCRATCH_TRAIN_DIR_NAME
HUB_MODEL_ID_SCRATCH = f"{NEW_MODEL_CARD_PREFIX}-{SCRATCH_TRAIN_DIR_NAME}"

# --- Hyperparameters for Scratch Training (NEEDS TUNING) ---
LEARNING_RATE_SCRATCH = 5e-5
WARMUP_STEPS_SCRATCH = 1000 # Warmup is usually step-based
WEIGHT_DECAY_SCRATCH = 0.01
GRADIENT_ACCUMULATION_STEPS = 1 # Set if using gradient accumulation





# load_dotenv(".env")
# EVAL_BATCH_SIZE = 2
# NUM_IMAGES_TO_TRAIN = 350000
# TRAIN_BATCH_SIZE = 3
# TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
# SAVE_LIMIT = 4
# SAVE_STEPS = 15000
# LOGGING_STEPS = 100
# IGNORE_ID = -100  # Pytorch ignore index when computing loss
# MAX_LENGTH = 512
# # GRID_ROOT = "path/to/grid_pickles"
# # BACKBONE_FILE_NAME = "model-8000.safetensors"
# # DATASET_ID = "katphlab/doclaynet-table"
# DATASET_NUM_PROC = 4
# DEVICE = "cuda"
# FREEZE_BACKBONE = False
# LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
# NEW_MODEL_CARD_PREFIX = "saeed11b95/multi-modal-prompt-detection"
# NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
# PRETRAIN_BACKBONE_ID = None  # Set to the repo id of the backbone model
# PRETRAIN_MODEL_ID = "/home/docanalysis/florence2-training/models"
# PROMPT = "<OD>"
# PUSH_TO_HUB = False
# RUN_NAME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# SCRATCH_TRAIN_DIR_NAME = "trained_scratch"
# # Save inside ./runs for better organization, or use Path(f"./{SCRATCH_TRAIN_DIR_NAME}") for root directory
# SAVE_DIRECTORY = Path("./runs")
# SCRATCH_OUTPUT_PATH = SAVE_DIRECTORY / SCRATCH_TRAIN_DIR_NAME

# HUB_MODEL_ID_SCRATCH = f"{NEW_MODEL_CARD_PREFIX}-{SCRATCH_TRAIN_DIR_NAME}"
# TRAIN_SPLIT_NAME = "train"
# VAL_SPLIT_NAME = "val"
# TRAIN_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/train'
# VAL_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/val'
# device = torch.device(DEVICE)

# LEARNING_RATE_SCRATCH = 5e-5 # Starting point, tune this!
# WARMUP_STEPS_SCRATCH = 1000  # Starting point, tune this!
# WEIGHT_DECAY_SCRATCH = 0.01 # Common value, tune this!c

ID2LABEL = {
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
PROMT_TO_CLASS = {"<AO>":"All",
            "<CAP>":"Caption",
            "<FTN": "Footnote",
            "<FRM>":"Formula",
            "<LST>":"ListItem",
            "<PGF>":"PageFooter",
            "<PGH>":"PageHeader",
            "<PIC>":"Picture",
            "<SHD>":"SectionHeader",
            "<TAB>":"Table",
            "<TXT>":"Text",
            "<TTL>":"Title",
            }
CLASS_TO_PROMPT = {v:k for k,v in PROMT_TO_CLASS.items()}
import random
#(This was originally in the code)
# def sample_class_strings(class_strings):
#     samp_prob = random.uniform(0, 1)
#     if samp_prob < 0.6:
#         return class_strings[0]
#     else:
#         return random.choice(class_strings[1:])

#(This I modified)        
def sample_class_strings(class_strings):
        return random.choice(class_strings[1:])    

def quant_bbox(width, height, boxes, category_ids):
    bins_w, bins_h = [1000, 1000]  # Quantization bins.
    size_per_bin_w = width / bins_w
    size_per_bin_h = height / bins_h

    CAT_ID_TO_USE = 9
    bbox_str_dict = {cat_id: ID2LABEL[cat_id] for cat_id in category_ids}
    for bbox, cat_id in zip(boxes, category_ids):
        if cat_id != CAT_ID_TO_USE:
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

        bbox_str_dict[
            cat_id
        ] += f"<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>"

    full_bbox_str = ""
    for bbox_str in bbox_str_dict.values():
        if "loc" not in bbox_str:
            continue
        full_bbox_str += bbox_str
    return full_bbox_str


def paddbbox_seq(bboxs):
    padd_value = [[0, 0, 0, 0]]
    max_len_seq = max([len(boxes) for boxes in bboxs])
    padded_bboxs = []
    for boxes in bboxs:
        padded_bboxs.append(boxes + padd_value * (max_len_seq - len(boxes)))
    return padded_bboxs


def padd_input_ids(input_ids):
    padd_value = [0]
    max_len_seq = max([len(inp_ids) for inp_ids in input_ids])
    padded_ids = []
    for inp_ids in input_ids:
        padded_ids.append(inp_ids + padd_value * (max_len_seq - len(inp_ids)))
    return padded_ids


def collate_fn(batch):
    label_texts = [sample_class_strings([example['bbox_str']] + example["class_strs"]) for example in batch]
    prompt_texts = [CLASS_TO_PROMPT[label_text.split("<")[0]] for label_text in label_texts]
    images = [example["image"] for example in batch]
    input_ids = padd_input_ids([example["input_ids"] for example in batch])
    subword_bboxs = paddbbox_seq([example["subword_bboxs"] for example in batch])
    grid_data = [{'input_ids': torch.tensor(input_id), 'bbox': torch.tensor(rescale_bboxs(torch.tensor(subword_bbox)))} for input_id, subword_bbox in zip(input_ids, subword_bboxs)]
    inputs = processor(
        images=images,
        text=prompt_texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
    )

    labels = processor.tokenizer(
        label_texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
    )["input_ids"]

    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
    return_data = {
        **inputs,
        "labels": labels,
        "grid_data": grid_data,
    }

    return return_data


def rescale_bboxs(bboxs):
    scale = 768 / 1025
    scaled_bboxs = []
    for box in bboxs:
        scaled_bboxs.append([x * scale for x in box])
    return scaled_bboxs

# def add_grid_train(data_example):
#     pickle_name = (
#         TRAIN_COCO.loadImgs([data_example["image_id"]])[0]["file_name"][:-3] + "pkl"
#     )
#     grid_path = os.path.join(GRID_ROOT, pickle_name)
#     grid = pickle.load(open(grid_path, "rb"))
#     data_example["input_ids"] = grid["input_ids"]
#     data_example["subword_bbox_list"] = rescale_bboxs(grid["subword_bbox_list"])
#     return data_example


# def add_grid_val(data_example):
#     pickle_name = (
#         VAL_COCO.loadImgs([data_example["image_id"]])[0]["file_name"][:-3] + "pkl"
#     )
#     grid_path = os.path.join(GRID_ROOT, pickle_name)
#     grid = pickle.load(open(grid_path, "rb"))
#     data_example["input_ids"] = grid["input_ids"]
#     data_example["subword_bbox_list"] = rescale_bboxs(grid["subword_bbox_list"])
#     return data_example

if __name__ == "__main__":

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    print(f"Ensuring output directory exists: {SCRATCH_OUTPUT_PATH}")
    SCRATCH_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load datasets *before* calculating steps per epoch
    print(f"Loading train dataset from: {TRAIN_DATA}")
    train_dataset = datasets.load_from_disk(TRAIN_DATA)
    print(f"Loading validation dataset from: {VAL_DATA}")
    val_dataset = datasets.load_from_disk(VAL_DATA)
    train_dataset_size = len(train_dataset)
    print(f"Train dataset size: {train_dataset_size}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- Calculate steps needed for epoch-based saving/evaluation ---
    # Note: Assumes single GPU or uses per_device_batch_size. For multi-GPU,
    # the Trainer internally handles the global batch size. If gradient_accumulation_steps > 1,
    # one optimization step happens every GRADIENT_ACCUMULATION_STEPS training steps.
    # Effective batch size per step = TRAIN_BATCH_SIZE * num_gpus * GRADIENT_ACCUMULATION_STEPS
    # We estimate based on per-device size and accumulation steps.
    steps_per_epoch = math.ceil(train_dataset_size / (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    save_steps_calculated = SAVE_EVERY_N_EPOCHS * steps_per_epoch
    eval_steps_calculated = EVAL_EVERY_N_EPOCHS * steps_per_epoch
    total_training_steps = NUM_EPOCHS_TO_TRAIN * steps_per_epoch
    print(f"Calculated steps per epoch: {steps_per_epoch}")
    print(f"Saving checkpoints every {SAVE_EVERY_N_EPOCHS} epochs (~{save_steps_calculated} steps)")
    print(f"Evaluating every {EVAL_EVERY_N_EPOCHS} epochs (~{eval_steps_calculated} steps)")
    print(f"Total training steps for {NUM_EPOCHS_TO_TRAIN} epochs: {total_training_steps}")
    # Adjust warmup steps if it exceeds total steps (unlikely here, but good practice)
    actual_warmup_steps = min(WARMUP_STEPS_SCRATCH, total_training_steps)
    print(f"Using warmup steps: {actual_warmup_steps}")
    # ----------------------------------------------------------------

    print(f"Loading processor from: {PRETRAIN_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)

    print(f"Loading configuration from: {PRETRAIN_MODEL_ID}")
    config = AutoConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    print("Model configuration loaded.")

    print("Initializing model from configuration (training from scratch)...")
    new_model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True)
    new_model.to(device)
    print(f"Model initialized with random weights and moved to {device}.")

    with open("model_sum.txt", "w", encoding="utf-8") as f:

        # 1. Full model print
        f.write("="*80 + "\n")
        f.write("FULL MODEL ARCHITECTURE\n")
        f.write("="*80 + "\n\n")
        f.write(str(new_model))
        f.write("\n\n\n")

        # 2. Model Summary
        f.write("="*80 + "\n")
        f.write("MODEL SUMMARY (using torchinfo.summary)\n")
        f.write("="*80 + "\n\n")
        try:
            model_summary = summary(new_model, depth=3, verbose=1)
            f.write(str(model_summary))
        except Exception as e:
            f.write(f"Could not generate summary: {str(e)}\n")
        f.write("\n\n\n")

        # 3. Top-Level Modules
        f.write("="*80 + "\n")
        f.write("TOP-LEVEL MODULES (named_children)\n")
        f.write("="*80 + "\n\n")
        for name, module in new_model.named_children():
            f.write(f"[{name}]\n")
            f.write(str(module).replace('\n', '\n    '))  # Indent each line
            f.write("\n\n")
        f.write("\n\n\n")

        # 4. All Parameters
        # 4. All Parameters
        f.write("="*80 + "\n")
        f.write("ALL PARAMETERS (name, shape, trainable)\n")
        f.write("="*80 + "\n\n")
        for name, param in new_model.named_parameters():
            trainable = "✅" if param.requires_grad else "❌"
            f.write(f"{name} : {list(param.shape)} | Trainable: {trainable}\n")
        f.write("\n\n")


        print("✅ Model summary nicely saved to model_sum.txt!")


    if FREEZE_BACKBONE:
         print("ERROR: FREEZE_BACKBONE is True!")
    else:
        num_params = sum(p.numel() for p in new_model.parameters())
        num_trainable_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
        print(f"All model parameters are trainable (FREEZE_BACKBONE=False).")
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable_params:,}")

    print("Setting up Training Arguments...")
    args = TrainingArguments(
        # --- Directory and Naming ---
        output_dir=str(SCRATCH_OUTPUT_PATH),
        run_name=SCRATCH_TRAIN_DIR_NAME,
        hub_model_id=HUB_MODEL_ID_SCRATCH,
        # overwrite_output_dir=True, # Uncomment to allow overwriting

        # --- Epoch and Step Control ---
        num_train_epochs=NUM_EPOCHS_TO_TRAIN, # Set total epochs
        # max_steps=-1, # Disable max_steps if num_train_epochs is set

        # --- Batching and Accumulation ---
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # --- Learning Rate and Optimization ---
        learning_rate=LEARNING_RATE_SCRATCH,
        warmup_steps=actual_warmup_steps, # Use calculated warmup steps
        weight_decay=WEIGHT_DECAY_SCRATCH,
        optim="adamw_torch",

        # --- Logging, Saving, Evaluation (Now based on calculated steps equivalent to N epochs) ---
        logging_steps=LOGGING_STEPS, # Keep step-based logging for frequency
        logging_strategy="steps",
        save_strategy="steps", # Use "steps" for saving every N epochs via calculation
        save_steps=save_steps_calculated, # Save every N epochs (calculated steps)
        eval_strategy="steps", # Use "steps" for evaluating every N epochs via calculation
        eval_steps=eval_steps_calculated, # Evaluate every N epochs (calculated steps)
        save_total_limit=SAVE_LIMIT, # Limit total checkpoints based on save frequency
        load_best_model_at_end=True, # Load best based on evaluation metric

        # --- Technical Settings ---
        fp16=True,
        dataloader_num_workers=6,
        label_names=["labels"],
        remove_unused_columns=False,

        # --- Reporting and Hub ---
        report_to="none" if not LOG_TO_WANDB else "wandb",
        push_to_hub=False,
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=new_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
        # Callbacks can still be used if needed, e.g., for custom logic
        # callbacks=[...]
    )

    # Start Training
    print(f"Starting training from scratch for {NUM_EPOCHS_TO_TRAIN} epochs...")
    print(f"Saving and evaluating every {SAVE_EVERY_N_EPOCHS} epochs (~{save_steps_calculated} steps).")
    if LOG_TO_WANDB:
        print(f"Logging to WandB run: {SCRATCH_TRAIN_DIR_NAME}")

    trainer.train(resume_from_checkpoint=None)

    # Post-Training
    print("Training finished.")
    best_ckpt_path = getattr(trainer.state, 'best_model_checkpoint', 'Not Found (check load_best_model_at_end)')
    print(f"Best checkpoint path: {best_ckpt_path}")

    # Push to Hub if enabled
    if args.push_to_hub:
        print(f"Pushing final best model and processor to Hub: {args.hub_model_id}")
        try:
            if best_ckpt_path and Path(best_ckpt_path).exists():
                 processor.save_pretrained(best_ckpt_path)
                 print(f"Processor saved to {best_ckpt_path}")
            else:
                 print(f"Best checkpoint path {best_ckpt_path} not found, saving processor to main output dir.")
                 processor.save_pretrained(args.output_dir)
            trainer.push_to_hub(commit_message=f"End of training for {NUM_EPOCHS_TO_TRAIN} epochs")
            print("Push to Hub successful.")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")

    if LOG_TO_WANDB:
        print("Finishing WandB run...")
        wandb.finish()

    print("Script execution finished.")