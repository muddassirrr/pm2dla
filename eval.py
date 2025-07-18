from datetime import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from fitz_utils import ProcessedDoc
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "<OD>"
CHECKPOINT_PATH = "katphlab/florence2-base-doclaynet-fintabnet-125000"
model_revision = "refs/pr/1"
print(CHECKPOINT_PATH)

config = AutoConfig.from_pretrained(
    CHECKPOINT_PATH, trust_remote_code=True, revision=model_revision
)
config.vision_config.model_type = "davit"
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.float16,
    revision=model_revision,
).to(device)
processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)


def run_example(task_prompt, image, max_new_tokens=128):
    prompt = task_prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].half().to(device),
        max_new_tokens=max_new_tokens,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed_answer


def plot_bbox(image, data):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(20, 20))

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data["bboxes"], data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
        )
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    # Remove the axis ticks and labels
    ax.axis("off")

    # Show the plot
    plt.show()


fname = Path("./91c2de50-dda8-4bcf-9683-6203e7068d88.pdf")
doc = ProcessedDoc(fname=fname)
start_time = datetime.now()
for page in tqdm(doc):
    img = page.get_opencv_img()
    image = Image.fromarray(img).convert("RGB")

    parsed_answer = run_example(PROMPT, image=image)
    bboxes = parsed_answer[PROMPT]["bboxes"]
    for bbox in bboxes:
        page.draw_rect(bbox, color=(1, 0, 0), fill=(1, 0, 0), fill_opacity=0.25)
    # print(parsed_answer)
    # plot_bbox(image, parsed_answer[PROMPT])

print(datetime.now() - start_time)
doc.save(f"{fname.stem}_eval.pdf")
doc.close()
