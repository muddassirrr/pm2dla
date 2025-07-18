import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb

load_dotenv(".env")

EVAL_BATCH_SIZE = 1
NUM_IMAGES_TO_TRAIN = 10000
TRAIN_BATCH_SIZE = 1
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 2
SAVE_STEPS = 1000
IGNORE_ID = -100  # Pytorch ignore index when computing loss
MAX_LENGTH = 512

DATASET_ID = "katphlab/doclaynet-table"
DATASET_NUM_PROC = 4
DEVICE = "cuda"
EARLY_STOPPING_THRESHOLD = 0.01
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
NEW_MODEL_CARD_PREFIX = "katphlab/florence2-base-distilled-doclaynet-table"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
PROMPT = "<OD>"
PUSH_TO_HUB = True
RUN_NAME = "distill-model-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
SAVE_DIRECTORY = Path("./runs")
STUDENT_MODEL_CARD = "microsoft/Florence-2-base-ft"
TEACHER_MODEL_CARD = "thewalnutaisg/florence2-large-doclaynet-70k"
TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"

device = torch.device(DEVICE)


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through student model
        labels = inputs["labels"]

        # Forward pass through teacher model
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        # Student forward pass
        outputs = model(**inputs)

        # Task-specific loss (e.g., object detection loss)
        task_loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=IGNORE_ID,
        )

        # Knowledge distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(outputs.logits / self.temperature, dim=-1),
            F.log_softmax(teacher_outputs.logits / self.temperature, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * (self.temperature**2)

        # Combined loss
        loss = self.alpha * task_loss + (1 - self.alpha) * kd_loss

        return (loss, outputs) if return_outputs else kd_loss


def collate_fn(batch):
    prompt_texts = [PROMPT for _ in batch]
    label_texts = [example["bbox_str"] for example in batch]
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
    # Load COCO dataset
    train_dataset = load_dataset(
        DATASET_ID, TRAIN_SPLIT_NAME, split=TRAIN_SPLIT_NAME, num_proc=DATASET_NUM_PROC
    )
    val_dataset = load_dataset(
        DATASET_ID, VAL_SPLIT_NAME, num_proc=DATASET_NUM_PROC, split=VAL_SPLIT_NAME
    )

    # Initialize models and processor
    processor = AutoProcessor.from_pretrained(
        STUDENT_MODEL_CARD, trust_remote_code=True
    )

    student_config = AutoConfig.from_pretrained(
        STUDENT_MODEL_CARD, trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_CARD, config=student_config, trust_remote_code=True
    )
    student_model.to(device)

    teacher_config = AutoConfig.from_pretrained(
        TEACHER_MODEL_CARD, trust_remote_code=True
    )
    teacher_config.vision_config.model_type = "davit"
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_CARD, config=teacher_config, trust_remote_code=True
    )
    # Freeze the teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(device)
    teacher_model.eval()

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

    trainer = DistillationTrainer(
        args=args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            )
        ],
        data_collator=collate_fn,
        model=student_model,
        tokenizer=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        teacher_model=teacher_model,
        alpha=0.5,
        temperature=2.0,
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
