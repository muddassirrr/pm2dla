from datasets import DatasetDict, load_from_disk
from huggingface_hub import login

train_dataset = load_from_disk("/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/train")
val_dataset = load_from_disk("/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/val")
test_dataset = load_from_disk("/home/muddassir/Desktop/LayoutAnalysis/code_modularized/dataset_florence2_td/test")

print("datasets loaded")

# Step 3: Combine into a DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Step 4: Push to Hugging Face Hub
dataset.push_to_hub("DoclayNet_with_grid")
