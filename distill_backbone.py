import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

import wandb
from models.modeling_florence2 import Florence2VisionModel
from utils import copy_weights

load_dotenv(".env")

EVAL_BATCH_SIZE = 1
NUM_IMAGES_TO_TRAIN = 1000
TRAIN_BATCH_SIZE = 1
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 1
SAVE_STEPS = 100

DATASET_CARD = "katphlab/doclaynet-table"
DATASET_NUM_PROC = 4
DEVICE = "cuda"
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
MAX_LENGTH = 512
PROMPT = "<OD>"
PUSH_TO_HUB = True
RUN_NAME = "distill-backbone-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
SAVE_DIRECTORY = Path("./runs")
STUDENT_MODEL_CARD = "microsoft/Florence-2-base-ft"
TEACHER_MODEL_CARD = "katphlab/florence2-large-doclaynet-70k"
TRAIN_SPLIT_NAME = "train"
SAVE_BACKBONE_CARD = f"katphlab/florence2-base-backbone-{TRAIN_STEPS}"

device = torch.device(DEVICE)


class StudentModelWithProjection(nn.Module, PyTorchModelHubMixin):
    def __init__(self, student_model):
        super().__init__()
        self.base_model = student_model
        self.projection = nn.Linear(1024, 2048)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        projected_features = self.projection(outputs)
        return projected_features


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through student model
        pixel_values = inputs.pop("pixel_values")
        student_features = model(pixel_values)

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_features = self.teacher_model(pixel_values)

        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(student_features / self.temperature, dim=-1),
            F.log_softmax(teacher_features / self.temperature, dim=-1),
            reduction="batchmean",
            log_target=True,
        ) * (self.temperature**2)

        return (kd_loss, student_features) if return_outputs else kd_loss


def collate_fn(batch):
    prompt_texts = [PROMPT for _ in batch]
    images = [example["image"] for example in batch]

    inputs = processor(
        images=images,
        text=prompt_texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
    )

    return {"pixel_values": inputs["pixel_values"]}


if __name__ == "__main__":
    # Load COCO dataset
    train_dataset = load_dataset(
        DATASET_CARD,
        TRAIN_SPLIT_NAME,
        split=TRAIN_SPLIT_NAME,
        num_proc=DATASET_NUM_PROC,
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
    student_backbone = Florence2VisionModel(student_config.vision_config)
    student_backbone_state_dict = copy_weights(
        student_model.state_dict(), student_backbone.state_dict()
    )
    student_backbone.load_state_dict(student_backbone_state_dict)
    del student_model
    student_model = StudentModelWithProjection(student_backbone)
    student_model.to(device)

    teacher_config = AutoConfig.from_pretrained(
        TEACHER_MODEL_CARD, trust_remote_code=True
    )
    teacher_config.vision_config.model_type = "davit"
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_CARD, config=teacher_config, trust_remote_code=True
    )
    teacher_backbone = Florence2VisionModel(teacher_config.vision_config)
    teacher_backbone_state_dict = copy_weights(
        teacher_model.state_dict(), teacher_backbone.state_dict()
    )
    teacher_backbone.load_state_dict(teacher_backbone_state_dict)
    del teacher_model
    teacher_backbone.eval()
    # Freeze the teacher model
    for param in teacher_backbone.parameters():
        param.requires_grad = False
    teacher_backbone.to(device)

    args = TrainingArguments(
        dataloader_num_workers=6,
        eval_steps=SAVE_STEPS,
        eval_strategy="steps",
        fp16=True,
        label_names=["labels"],
        learning_rate=1e-6,
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
    )

    trainer = DistillationTrainer(
        args=args,
        data_collator=collate_fn,
        model=student_model,
        tokenizer=processor,
        train_dataset=train_dataset,
        teacher_model=teacher_backbone,
        alpha=0.5,
        temperature=2.0,
    )

    trainer.train()

    if PUSH_TO_HUB:
        trainer.push_to_hub(hub_model_id=SAVE_BACKBONE_CARD, private=True)

    if LOG_TO_WANDB:
        wandb.finish()
    print("Finished")
