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

import wandb
from utils import copy_weights

load_dotenv(".env")

EVAL_BATCH_SIZE = 2
NUM_IMAGES_TO_TRAIN = 50000
TRAIN_BATCH_SIZE = 4
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 4
SAVE_STEPS = 1000
IGNORE_ID = -100  # Pytorch ignore index when computing loss
MAX_LENGTH = 512

BACKBONE_FILE_NAME = "model-8000.safetensors"
DATASET_ID = "katphlab/doclaynet-table"
DATASET_NUM_PROC = 4
DEVICE = "cuda"
FREEZE_BACKBONE = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
NEW_MODEL_CARD_PREFIX = "katphlab/florence2-base-distilledbackbone"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
PRETRAIN_BACKBONE_ID = None  # Set to the repo id of the backbone model
PRETRAIN_MODEL_ID = "microsoft/Florence-2-base-ft"
PROMPT = "<OD>"
PUSH_TO_HUB = False
RUN_NAME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
SAVE_DIRECTORY = Path("./runs")
TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"

device = torch.device(DEVICE)

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

        bbox_str_dict[cat_id] += (
            f"<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>"
        )

    full_bbox_str = ""
    for bbox_str in bbox_str_dict.values():
        if "loc" not in bbox_str:
            continue
        full_bbox_str += bbox_str
    return full_bbox_str


def collate_fn(batch):
    prompt_texts = [PROMPT for _ in batch]
    label_texts = [
        quant_bbox(
            example["width"],
            example["height"],
            example["boxes"],
            example["category_ids"],
        )
        for example in batch
    ]
    images = [example["image"] for example in batch]

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
        return_token_type_ids=False,  # no need to set this to True since BART does not use token type ids
    )["input_ids"]

    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
    return_data = {**inputs, "labels": labels}

    return return_data


if __name__ == "__main__":
    train_dataset = datasets.load_dataset(
        DATASET_ID, TRAIN_SPLIT_NAME, num_proc=DATASET_NUM_PROC, split=TRAIN_SPLIT_NAME
    )
    val_dataset = datasets.load_dataset(
        DATASET_ID, VAL_SPLIT_NAME, num_proc=DATASET_NUM_PROC, split=VAL_SPLIT_NAME
    )

    processor = AutoProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config.vision_config.model_type = "davit"
    new_model = AutoModelForCausalLM.from_pretrained(
        PRETRAIN_MODEL_ID, trust_remote_code=True, config=config
    )

    if PRETRAIN_BACKBONE_ID:
        local_dir = SAVE_DIRECTORY / "backbone"
        file_path = local_dir / BACKBONE_FILE_NAME
        hf_hub_download(PRETRAIN_BACKBONE_ID, BACKBONE_FILE_NAME, local_dir=local_dir)
        ckpt_state_dict = load_file(file_path)
        vision_state_dict = copy_weights(ckpt_state_dict, new_model.state_dict())
        new_model.load_state_dict(vision_state_dict)

    if FREEZE_BACKBONE:
        for param in new_model.vision_tower.parameters():
            param.requires_grad = False

    args = TrainingArguments(
        dataloader_num_workers=6,
        eval_steps=SAVE_STEPS,
        eval_strategy="steps",
        fp16=True,
        label_names=["labels"],
        learning_rate=1e-6,
        load_best_model_at_end=True,  # we will manually push model to the hub at the end of training
        logging_steps=SAVE_STEPS,
        logging_strategy="steps",
        max_steps=TRAIN_STEPS,
        output_dir=f"{SAVE_DIRECTORY}/{RUN_NAME}",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        push_to_hub=False,
        remove_unused_columns=False,  # needed for data collator
        report_to="none" if not LOG_TO_WANDB else "wandb",
        run_name=RUN_NAME,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_LIMIT,
        hub_model_id=NEW_MODEL_CARD,
    )

    trainer = Trainer(
        args=args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.001
            )
        ],
        data_collator=collate_fn,
        model=new_model,
        tokenizer=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train(resume_from_checkpoint=None)
    best_ckpt_path = trainer.state.best_model_checkpoint
    print(best_ckpt_path)
    if PUSH_TO_HUB:
        processor.push_to_hub(NEW_MODEL_CARD, private=True)
        trainer.push_to_hub(NEW_MODEL_CARD, private=True)

    if LOG_TO_WANDB:
        wandb.finish()
    print("Finished")
