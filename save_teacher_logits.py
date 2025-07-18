import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

load_dotenv()

TEACHER_MODEL_CARD = os.environ["CUSTOM_MODEL_CARD"]
STUDENT_MODEL_CARD = "microsoft/Florence-2-base-ft"
DATASET_CARD = "katphlab/doclaynet-table"
PROMPT = "<OD>"
IGNORE_ID = -100  # Pytorch ignore index when computing loss
MAX_LENGTH = 512
DATA_SPLIT = "val"
RUN_NAME = "distillv2"
OUTPUT_DIR = "./runs"

# Initialize models and processor
processor = AutoProcessor.from_pretrained(STUDENT_MODEL_CARD, trust_remote_code=True)
config = AutoConfig.from_pretrained(TEACHER_MODEL_CARD, trust_remote_code=True)
config.vision_config.model_type = "davit"
teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_CARD, trust_remote_code=True, config=config
)
# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)

# Freeze the teacher model
for param in teacher_model.parameters():
    param.requires_grad = False

# Load COCO dataset
dataset = load_dataset(DATASET_CARD, DATA_SPLIT, split=DATA_SPLIT, num_proc=12)


def compute_cpm_for_k(logits, k_values):
    """
    Compute the Cumulative Probability Mass (CPM) for different k values.

    Args:
    logits (torch.Tensor): The full logits from the teacher model.
    k_values (list): List of k values to evaluate.

    Returns:
    dict: A dictionary with k values as keys and mean CPM as values.
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)

    cpm_dict = {}
    for k in k_values:
        top_k_probs = sorted_probs[:, :, :k]
        cpm = top_k_probs.sum(dim=-1).mean().item()
        cpm_dict[k] = cpm

    return cpm_dict


def select_optimal_k(cpm_dict, threshold=0.95):
    """
    Select the smallest k that achieves a CPM above the threshold.

    Args:
    cpm_dict (dict): Dictionary with k values as keys and mean CPM as values.
    threshold (float): The desired CPM threshold.

    Returns:
    int: The optimal k value.
    """
    for k, cpm in sorted(cpm_dict.items()):
        if cpm >= threshold:
            return k
    return max(cpm_dict.keys())  # If no k meets the threshold, return the largest k


def reconstruct_logits_from_topk(top_k_logits, top_k_indices, vocab_size):
    # Convert to tensors if they're lists
    top_k_logits = (
        torch.tensor(top_k_logits) if isinstance(top_k_logits, list) else top_k_logits
    )
    top_k_indices = (
        torch.tensor(top_k_indices)
        if isinstance(top_k_indices, list)
        else top_k_indices
    )

    # Ensure tensors are of type float and long respectively
    top_k_logits = top_k_logits.float()
    top_k_indices = top_k_indices.long()

    # Get the shape of the original logits
    seq_len, k = top_k_indices.shape

    # Create a tensor with the default logit value
    reconstructed_logits = torch.full((seq_len, vocab_size), 0.0)

    # Use scatter to place the top-k logits in the correct positions
    reconstructed_logits.scatter_(-1, top_k_indices, top_k_logits)

    return reconstructed_logits


# Create DataLoader
def preprocess_function(examples):
    prompt_texts = [PROMPT] * len(examples["image"])

    inputs = processor(
        images=examples["image"],
        text=prompt_texts,
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
    )

    labels = processor.tokenizer(
        examples["bbox_str"],
        return_tensors="pt",
        padding="longest",
        max_length=MAX_LENGTH,
        return_token_type_ids=False,
    )["input_ids"]

    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_ID
    # No need to remove batch dimension as we're processing in batches
    inputs["labels"] = labels

    # Move all inputs to CUDA
    inputs_cuda = {k: v.to(device) for k, v in inputs.items()}

    # Compute teacher logits
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs_cuda)
        teacher_logits = teacher_outputs.logits

    k = 10  # You can adjust this value
    top_k_logits, top_k_indices = torch.topk(teacher_logits.cpu(), k, dim=-1)
    inputs["top_k_logits"] = top_k_logits
    inputs["top_k_indices"] = top_k_indices

    return inputs


logit_dataset = dataset.map(preprocess_function, batched=True, batch_size=2)
logit_dataset.save_to_disk(f"./data/{DATASET_CARD}_{DATA_SPLIT}_logits")
