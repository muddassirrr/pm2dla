import os
from datetime import datetime
from pathlib import Path
import random
import json
import datasets
import torch
from torchinfo import summary
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

# Config
EVAL_BATCH_SIZE = 2
NUM_IMAGES_TO_TRAIN = 1400000
TRAIN_BATCH_SIZE = 3
TRAIN_STEPS = NUM_IMAGES_TO_TRAIN // TRAIN_BATCH_SIZE
SAVE_LIMIT = 4
SAVE_STEPS = 15000
LOGGING_STEPS = 100
IGNORE_ID = -100
MAX_LENGTH = 512
DEVICE = "cuda"
FREEZE_BACKBONE = False
LOG_TO_WANDB = os.getenv("LOG_TO_WANDB", "0") != "0"
NEW_MODEL_CARD_PREFIX = "saeed11b95/multi-modal-prompt-detection"
NEW_MODEL_CARD = f"{NEW_MODEL_CARD_PREFIX}-{TRAIN_STEPS}"
PRETRAIN_BACKBONE_ID = None
PRETRAIN_MODEL_ID = "/home/docanalysis/florence2-training/models"
PUSH_TO_HUB = False
RUN_NAME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
SAVE_DIRECTORY = Path("/opt/doclay")
TRAIN_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/train'
VAL_DATA = '/home/docanalysis/florence2-training/dataset_florence2_td/val'

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

device = torch.device(DEVICE)

ID2LABEL = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "ListItem",
    5: "PageFooter", 6: "PageHeader", 7: "Picture", 8: "SectionHeader",
    9: "Table", 10: "Text", 11: "Title"
}

CLASS_WEIGHTS = {
    'Caption': 0.0849, 'Footnote': 0.2881, 'Formula': 0.0767, 'ListItem': 0.0100,
    'PageFooter': 0.0265, 'PageHeader': 0.0339, 'Picture': 0.0410,
    'SectionHeader': 0.0137, 'Table': 0.0540, 'Text': 0.0038, 'Title': 0.3674,
}

PROMPT_TO_CLASS = {
    "<CAP>": "Caption", "<FTN>": "Footnote", "<FRM>": "Formula",
    "<LST>": "ListItem", "<PGF>": "PageFooter", "<PGH>": "PageHeader",
    "<PIC>": "Picture", "<SHD>": "SectionHeader", "<TAB>": "Table",
    "<TXT>": "Text", "<TTL>": "Title",
}
CLASS_TO_PROMPT = {v: k for k, v in PROMPT_TO_CLASS.items()}

# Sampling strategy
def sample_balanced_class(class_strings, class_weights):
    if not class_strings:
        return None
    available_classes = [s.split('<')[0] for s in class_strings]
    weights_for_available = [class_weights.get(name, 0) for name in available_classes]
    if sum(weights_for_available) == 0:
        print("No valid classes found, returning a random class.")
        return random.choice(class_strings)

    return random.choices(class_strings, weights=weights_for_available, k=1)[0]

def padd_input_ids(input_ids):
    max_len_seq = max(len(ids) for ids in input_ids)
    return [ids + [0] * (max_len_seq - len(ids)) for ids in input_ids]

def paddbbox_seq(bboxs):
    max_len_seq = max(len(b) for b in bboxs)
    return [b + [[0,0,0,0]] * (max_len_seq - len(b)) for b in bboxs]

def rescale_bboxs(bboxs):
    scale = 768 / 1025
    return [[x * scale for x in box] for box in bboxs]

def collate_fn(batch):
    label_texts = [sample_balanced_class(example["class_strs"], CLASS_WEIGHTS) for example in batch]
    prompt_texts = [CLASS_TO_PROMPT[label.split("<")[0]] for label in label_texts]
    images = [example["image"] for example in batch]
    input_ids = padd_input_ids([example["input_ids"] for example in batch])
    subword_bboxs = paddbbox_seq([example["subword_bboxs"] for example in batch])
    grid_data = [
        {"input_ids": torch.tensor(inp), "bbox": torch.tensor(rescale_bboxs(torch.tensor(bbox)))}
        for inp, bbox in zip(input_ids, subword_bboxs)
    ]
    inputs = processor(
        images=images, text=prompt_texts, return_tensors="pt",
        padding="longest", max_length=MAX_LENGTH,
    )
    labels = processor.tokenizer(
        label_texts, return_tensors="pt", padding="longest",
        max_length=MAX_LENGTH, return_token_type_ids=False
    )["input_ids"]
    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
    return {**inputs, "labels": labels, "grid_data": grid_data}

# Custom Trainer for logging
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_history_file = SAVE_DIRECTORY / f"training_logs_{RUN_NAME}.json"
        self.training_logs = []

    def log(self, logs, iterator_start_time=None):
        super().log(logs, iterator_start_time)
        log_entry = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "timestamp": datetime.now().isoformat(),
            **logs
        }
        self.training_logs.append(log_entry)
        if self.state.global_step % LOGGING_STEPS == 0:
            self.save_logs_to_file()

    def save_logs_to_file(self):
        with open(self.log_history_file, 'w') as f:
            json.dump(self.training_logs, f, indent=2)

if __name__ == "__main__":
    train_dataset = datasets.load_from_disk(TRAIN_DATA)
    val_dataset = datasets.load_from_disk(VAL_DATA)

    processor = AutoProcessor.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True)
    config.vision_config.model_type = "davit"
    model = AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL_ID, trust_remote_code=True, config=config)

    if PRETRAIN_BACKBONE_ID:
        local_dir = SAVE_DIRECTORY / "backbone"
        file_path = local_dir / BACKBONE_FILE_NAME
        hf_hub_download(PRETRAIN_BACKBONE_ID, BACKBONE_FILE_NAME, local_dir=local_dir)
        ckpt_state_dict = load_file(file_path)
        vision_state_dict = copy_weights(ckpt_state_dict, model.state_dict())
        model.load_state_dict(vision_state_dict)

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
        push_to_hub=False,
        remove_unused_columns=False,
        report_to="none" if not LOG_TO_WANDB else "wandb",
        run_name=RUN_NAME,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_LIMIT,
        hub_model_id=NEW_MODEL_CARD,
    )

    trainer = CustomTrainer(
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=310, early_stopping_threshold=0.0001)],
        data_collator=collate_fn,
        model=model,
        tokenizer=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train(resume_from_checkpoint=None)
    trainer.save_logs_to_file()

    best_ckpt_path = trainer.state.best_model_checkpoint
    print("Best checkpoint:", best_ckpt_path)

    if PUSH_TO_HUB:
        processor.push_to_hub(NEW_MODEL_CARD, private=True)
        trainer.push_to_hub(NEW_MODEL_CARD, private=True)

    if LOG_TO_WANDB:
        wandb.finish()

    print("✅ Training completed.")
