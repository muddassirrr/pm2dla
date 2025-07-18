from datetime import datetime
from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
load_dotenv()
from models.modeling_florence2 import Florence2ForConditionalGeneration
from models.processing_florence2 import Florence2Processor
device = torch.device("cuda")

def run_example(task_prompt, example, max_new_tokens=128, input_text: str = None):
   prompt = task_prompt
   if input_text is not None:
       prompt += input_text
   image = example["image"]
   input_ids = torch.tensor(example['input_ids'])
   bbox = torch.tensor(example['subword_bboxs'])
   inputs = processor_f(text=prompt,images=image, return_tensors="pt")
   inputs = {**inputs, "grid_data":[{"input_ids": input_ids, "bbox": bbox}]}
   generated_ids = model_f.generate(
       input_ids=inputs["input_ids"].to(device),
       pixel_values = inputs["pixel_values"].half().to(device),
       grid_data = inputs['grid_data'],
       max_new_tokens=max_new_tokens,
       early_stopping=False,
       do_sample=False,
       num_beams=3,)
  
   generated_text = processor_f.batch_decode(generated_ids, skip_special_tokens=False)[0]
   parsed_answer = processor_f.post_process_generation(
       generated_text, task=task_prompt, image_size=(image.width, image.height)
   )
   return parsed_answer

def parse_location_tokens(text, expected_class, image_size):
    """
    Parse location tokens from text like 'figure<loc_87><loc_92><loc_910><loc_419>'
    Returns list of bounding boxes in [x1, y1, x2, y2] format
    """
    import re
    
    # Extract the class name and location tokens
    # Pattern to match sequences of location tokens
    loc_pattern = r'<loc_(\d+)>'
    matches = re.findall(loc_pattern, text)
    
    if len(matches) % 4 != 0:
        print(f"Warning: Invalid number of location tokens ({len(matches)}) in: {text}")
        return []
    
    boxes = []
    # Group location tokens into sets of 4 (x1, y1, x2, y2)
    for i in range(0, len(matches), 4):
        if i + 3 < len(matches):
            # Convert location tokens to coordinates
            # Assuming location tokens are in 0-999 range and need scaling
            x1, y1, x2, y2 = map(int, matches[i:i+4])
            
            # Scale to image dimensions (assuming 0-999 coordinate system)
            image_width, image_height = image_size
            x1_scaled = (x1 / 999.0) * image_width
            y1_scaled = (y1 / 999.0) * image_height
            x2_scaled = (x2 / 999.0) * image_width
            y2_scaled = (y2 / 999.0) * image_height
            
            boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
    
    return boxes

def plot_bounding_boxes(image,bboxs, labels, figsize=(10, 10)):
   fig, ax = plt.subplots(figsize=figsize)
   ax.imshow(image)
   for box, label in zip(bboxs, labels):
       rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=0.5, edgecolor='r', facecolor='g',alpha=0.3)
       #add class name
       ax.text(box[0], box[1],label, fontsize=12, color='red')
       ax.add_patch(rect)
   plt.show()

# Updated configuration for Publaynet
CHECKPOINT_PATH ="/opt/publaynet/runs/2025-06-11-15-17-05/checkpoint-465000"
DATASET_PATH = "/opt/publaynet/Publaynet/dataset_florence2_publaynet_simple/test"

print(CHECKPOINT_PATH)
config_f = AutoConfig.from_pretrained(
   CHECKPOINT_PATH, trust_remote_code=True
)
config_f.vision_config.model_type = "davit"
config_f.vision_config.in_chans = 3
model_f = (
   AutoModelForCausalLM.from_pretrained(
       CHECKPOINT_PATH,
       trust_remote_code=True,
       config=config_f,
       torch_dtype=torch.float16,
       ignore_mismatched_sizes=True,
   )
   .to(device)
   .eval()
)
processor_f = AutoProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)

# Updated mappings for Publaynet
ID2LABEL = {
   1: "text",
   2: "title",
   3: "list",
   4: "table",
   5: "figure",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Updated prompt mappings for PubLayNet classes
PROMPT_TO_CLASS = {
   "<TXT>": "text",
   "<TTL>": "title",
   "<LST>": "list",
   "<TAB>": "table",
   "<FIG>": "figure",
}
CLASS_TO_PROMPT = {v: k for k, v in PROMPT_TO_CLASS.items()}

from datasets import load_from_disk
dataset = load_from_disk(DATASET_PATH)
          
import torch
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time # Optional: for timing inference

# --- Evaluation Configuration ---
IOU_THRESHOLD = 0.5 # IoU threshold for TP/FP/FN determination
# Limit the number of samples for testing (set to None to use the full dataset)
NUM_SAMPLES = None # e.g., 100

# --- Helper Functions ---

def calculate_iou(boxA, boxB):
  """Calculates Intersection over Union (IoU) between two bounding boxes.
  Boxes are expected in [x_min, y_min, x_max, y_max] format.
  """
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  interArea = max(0, xB - xA) * max(0, yB - yA)
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  unionArea = boxAArea + boxBArea - interArea + 1e-6 # Epsilon for stability
  iou = interArea / unionArea
  return iou

def evaluate_layout_model(dataset, prompt_map, label_map_rev, iou_threshold=0.5, num_samples=None):
  """
  Evaluates the layout model, calculating TP, FP, FN, and sum of IoUs for TPs.
  Only prompts for classes present in the ground truth of each image.

  Args:
      dataset: The dataset object (Hugging Face dataset).
      prompt_map: Dictionary mapping prompt strings to label strings (e.g., "<TXT>": "text").
      label_map_rev: Dictionary mapping label strings to prompt strings (e.g., "text": "<TXT>").
      iou_threshold: The IoU threshold for matching predictions to ground truth.
      num_samples: Max number of samples to evaluate on (or None for all).

  Returns:
      A tuple: (results, all_tp_iou_sum)
        - results: Dictionary containing evaluation results per class
                   (TP, FP, FN, tp_iou_sum).
        - all_tp_iou_sum: Float, the total sum of IoU values for all True Positives across all classes.
  """
  # Initialize results dict, adding 'tp_iou_sum'
  results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tp_iou_sum': 0.0})
  all_tp_iou_sum = 0.0 # Accumulator for overall average IoU calculation
  total_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
  processed_samples = 0

  iterator = tqdm(dataset.select(range(total_samples)), desc="Evaluating Samples") if num_samples else tqdm(dataset, desc="Evaluating Samples")

  for example in iterator:
      try:
          # Load image from path for Publaynet dataset
          image_path = example['image_path']
          try:
              image = Image.open(image_path).convert('RGB')
          except Exception as e:
              print(f"Error loading image from {image_path}: {e}")
              continue
             
          gt_boxes_all = example.get('boxes', [])
          gt_category_ids = example.get('category_ids', [])

          # --- Data Validation and Preparation ---
          if not isinstance(gt_boxes_all, list) or (gt_boxes_all and not isinstance(gt_boxes_all[0], (list, torch.Tensor))):
               print(f"Warning: Unexpected format for 'boxes' in sample. Skipping sample.")
               continue
          if not isinstance(gt_category_ids, list): # Allow empty list
              gt_category_ids = []
          elif gt_category_ids and not isinstance(gt_category_ids[0], int):
               if len(gt_category_ids) == 1 and isinstance(gt_category_ids[0], list):
                   gt_category_ids = gt_category_ids[0] # Handle nested list case
               elif not all(isinstance(item, int) for item in gt_category_ids):
                    print(f"Warning: Unexpected format for 'category_ids' in sample. Skipping sample.")
                    continue

          gt_labels_present = set()
          gt_by_class = defaultdict(list)
          for idx, cat_id in enumerate(gt_category_ids):
              if cat_id in ID2LABEL:
                  label = ID2LABEL[cat_id]
                  gt_labels_present.add(label)
                  box = gt_boxes_all[idx]
                  gt_by_class[label].append(box.tolist() if isinstance(box, torch.Tensor) else box)
              # else: Silently ignore unknown category IDs

          prompts_for_this_image = [label_map_rev[label] for label in gt_labels_present if label in label_map_rev]

          if not prompts_for_this_image:
              processed_samples+=1
              continue
          # --- End Data Validation ---

          # --- Run Inference ONLY for relevant prompts ---
          for prompt in prompts_for_this_image:
              class_label = prompt_map.get(prompt)
              if not class_label: continue

              # --- Run Inference ---
              try:
                  inference_example = {
                      'image': image,
                      'input_ids': example.get('input_ids'),
                      'subword_bboxs': example.get('subword_bboxs')
                  }
                  inference_output = run_example(task_prompt=prompt, example=inference_example)
                  
                  # Initialize default values
                  filtered_pred_boxes = []
                  
                  # Handle the different output formats
                  if isinstance(inference_output, dict) and prompt in inference_output:
                      result = inference_output[prompt]
                      
                      # Check if result is a string (like for <FIG>) or dict (like for other classes)
                      if isinstance(result, str):
                          # Parse the location tokens from the string
                          filtered_pred_boxes = parse_location_tokens(result, class_label, (image.width, image.height))
                      elif isinstance(result, dict):
                          # Normal case - dict with bboxes and labels
                          pred_boxes = result.get('bboxes', [])
                          pred_labels = result.get('labels', [])
                          filtered_pred_boxes = [box for box, lbl in zip(pred_boxes, pred_labels) if lbl == class_label]
                      else:
                          print(f"Warning: Unexpected result format for {prompt}: {type(result)}")
                  else:
                      print(f"Warning: Unexpected inference_output format for {prompt}")
                      
              except Exception as e:
                  print(f"Error during inference for sample, prompt {prompt}: {e}")
                  filtered_pred_boxes = []
              # --- End Inference ---

              # --- Match Predictions to Ground Truth ---
              gt_boxes_cls = gt_by_class[class_label]
              num_gt = len(gt_boxes_cls)
              num_pred = len(filtered_pred_boxes)

              if num_gt == 0 and num_pred == 0: continue

              gt_matched = [False] * num_gt
              pred_matched = [False] * num_pred
              tp = 0
              current_tp_iou_sum = 0.0 # Sum IoUs for TPs *in this image/class*

              if num_pred > 0 and num_gt > 0:
                  iou_matrix = np.zeros((num_pred, num_gt))
                  valid_preds = [] # Store indices of valid predictions
                  for p_idx, p_box in enumerate(filtered_pred_boxes):
                      if not isinstance(p_box, list) or len(p_box) != 4:
                          print(f"Warning: Invalid prediction box format skipped: {p_box}")
                          continue
                      valid_preds.append(p_idx) # Track valid prediction index
                      valid_gt = [] # Store indices of valid ground truths for this pred
                      for g_idx, g_box in enumerate(gt_boxes_cls):
                          if not isinstance(g_box, list) or len(g_box) != 4:
                              # This check might be redundant if GT is clean, but safe
                              print(f"Warning: Invalid ground truth box format skipped: {g_box}")
                              continue
                          valid_gt.append(g_idx) # Track valid GT index
                          # Calculate IoU only between valid boxes
                          iou_matrix[p_idx, g_idx] = calculate_iou(p_box, g_box)

                  # Ensure we only consider valid boxes in matching
                  if valid_preds and valid_gt:
                       # Consider only the submatrix of valid preds/GTs for matching
                       valid_iou_matrix = iou_matrix[np.ix_(valid_preds, valid_gt)]
                       potential_matches = np.argwhere(valid_iou_matrix >= iou_threshold)

                       if potential_matches.size > 0:
                          # Map indices back to original filtered_pred_boxes and gt_boxes_cls
                          original_p_indices = [valid_preds[i] for i in potential_matches[:, 0]]
                          original_g_indices = [valid_gt[i] for i in potential_matches[:, 1]]

                          iou_values = valid_iou_matrix[potential_matches[:, 0], potential_matches[:, 1]]
                          sorted_indices = np.argsort(iou_values)[::-1] # Sort descending

                          for idx in sorted_indices:
                              p_idx = original_p_indices[idx] # Original index in filtered_pred_boxes
                              g_idx = original_g_indices[idx] # Original index in gt_boxes_cls

                              # Check if boxes (by original index) are already matched
                              if not pred_matched[p_idx] and not gt_matched[g_idx]:
                                  tp += 1
                                  pred_matched[p_idx] = True
                                  gt_matched[g_idx] = True
                                  # Accumulate IoU for this specific TP match
                                  iou_value = iou_matrix[p_idx, g_idx] # Get the actual IoU
                                  current_tp_iou_sum += iou_value

              # Calculate FP, FN for this class/image
              fp = sum(1 for p_idx, p_box in enumerate(filtered_pred_boxes) if not pred_matched[p_idx] and isinstance(p_box, list) and len(p_box) == 4) # Count only valid unmatched preds as FP
              fn = num_gt - sum(gt_matched)

              # Accumulate results for the class across all images
              results[class_label]['tp'] += tp
              results[class_label]['fp'] += fp
              results[class_label]['fn'] += fn
              results[class_label]['tp_iou_sum'] += current_tp_iou_sum # Add sum for this image/class

          processed_samples += 1 # Increment after successfully processing all prompts for the image

      except Exception as e:
          print(f"Fatal error processing sample (Index maybe unreliable if dataset shuffled): {e}")
          # Optionally log sample ID if available: example.get('image_id', 'Unknown')
          continue # Move to the next sample

  # --- Calculate total IoU sum after processing all samples ---
  # Sum up the per-class IoU sums collected in the results dict
  all_tp_iou_sum = sum(counts.get('tp_iou_sum', 0.0) for counts in results.values())

  print(f"\nProcessed {processed_samples} samples.")
  if processed_samples == 0:
       print("Warning: No samples were processed successfully.")

  return results, all_tp_iou_sum # Return both results dict and total IoU sum

def calculate_metrics(results, all_tp_iou_sum):
  """Calculates Precision, Recall, F1, and Avg IoU from TP, FP, FN counts."""
  metrics = {}
  all_tp, all_fp, all_fn = 0, 0, 0

  valid_labels = [label for label, counts in results.items() if counts['tp'] > 0 or counts['fp'] > 0 or counts['fn'] > 0]

  if not valid_labels:
      print("\nNo valid results found to calculate metrics.")
      return {}

  print("\n--- Per-Class Metrics ---")
  class_precisions = []
  class_recalls = []
  class_f1s = []
  class_avg_ious = [] # To calculate overall average IoU correctly later if needed

  for label in sorted(valid_labels): # Sort for consistent output
      counts = results[label]
      tp = counts['tp']
      fp = counts['fp']
      fn = counts['fn']
      tp_iou_sum = counts.get('tp_iou_sum', 0.0) # Get the sum of IoUs for TPs

      all_tp += tp
      all_fp += fp
      all_fn += fn

      # Calculate P, R, F1
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

      # Calculate Average IoU for TPs of this class
      avg_iou_class = tp_iou_sum / tp if tp > 0 else 0.0
      if tp > 0:
          class_avg_ious.append(avg_iou_class) # Store for potential macro avg IoU

      metrics[label] = {
          'precision': precision, 'recall': recall, 'f1': f1,
          'avg_iou': avg_iou_class, # Add avg IoU here
          'tp': tp, 'fp': fp, 'fn': fn
      }
      class_precisions.append(precision)
      class_recalls.append(recall)
      class_f1s.append(f1)

      # Print metrics for the class
      print(f"{label}:")
      print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
      print(f"  Precision: {precision:.4f}")
      print(f"  Recall:    {recall:.4f}")
      print(f"  F1-Score:  {f1:.4f}")
      print(f"  Avg TP IoU:{avg_iou_class:.4f}") # Print avg IoU

  # --- Overall Metrics (Micro Average) ---
  # Micro P, R, F1 are calculated based on total TP, FP, FN
  micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
  micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
  micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

  # Overall Average IoU is calculated based on total sum of TP IoUs and total TPs
  overall_avg_iou = all_tp_iou_sum / all_tp if all_tp > 0 else 0.0

  metrics['overall_micro'] = {
      'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1,
      'avg_iou': overall_avg_iou, # Add overall avg IoU here
      'tp': all_tp, 'fp': all_fp, 'fn': all_fn
  }
  print("\n--- Overall Metrics (Micro Average) ---")
  print(f"Total TP: {all_tp}, Total FP: {all_fp}, Total FN: {all_fn}")
  print(f"Precision: {micro_precision:.4f}")
  print(f"Recall:    {micro_recall:.4f}")
  print(f"F1-Score:  {micro_f1:.4f}")
  print(f"Avg TP IoU:{overall_avg_iou:.4f}") # Print overall avg IoU

  # --- Overall Metrics (Macro Average) ---
  num_classes_evaluated = len(valid_labels)
  if num_classes_evaluated > 0:
      macro_precision = sum(class_precisions) / num_classes_evaluated
      macro_recall = sum(class_recalls) / num_classes_evaluated
      if (macro_precision + macro_recall) > 0:
          macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
      else:
          macro_f1 = 0.0
      # Optional: Macro Average IoU (average of per-class average IoUs)
      # macro_avg_iou = sum(class_avg_ious) / len(class_avg_ious) if class_avg_ious else 0.0
  else:
      macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
      # macro_avg_iou = 0.0

  metrics['overall_macro'] = {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1}
  print("\n--- Overall Metrics (Macro Average) ---")
  print(f"Averaged over {num_classes_evaluated} classes: {', '.join(sorted(valid_labels))}")
  print(f"Precision: {macro_precision:.4f}")
  print(f"Recall:    {macro_recall:.4f}")
  print(f"F1-Score:  {macro_f1:.4f}")
  # print(f"Avg TP IoU:{macro_avg_iou:.4f}") # Optional: print macro avg IoU

  return metrics

# --- Main Execution ---
if __name__ == "__main__":
  # --- Define Mappings (Essential!) ---
  # Filter mappings based on labels actually in ID2LABEL
  filtered_prompt_to_label = {p: l for p, l in PROMPT_TO_CLASS.items() if l in ID2LABEL.values()}
  label_to_prompt_map = {v: k for k, v in filtered_prompt_to_label.items()}

  print(f"Using {len(filtered_prompt_to_label)} mappings based on ID2LABEL.")
  print("Mappings:", filtered_prompt_to_label)

  if not filtered_prompt_to_label:
      print("Error: No valid prompt-to-label mappings found. Check ID2LABEL and PROMPT_TO_CLASS.")
      import sys
      sys.exit(1)
  else:
      # --- Ensure dataset is loaded ---
      if 'dataset' not in globals():
          print("Error: 'dataset' is not defined. Please load your dataset.")
          import sys
          sys.exit(1)

      print(f"Dataset loaded with {len(dataset)} samples")
      print("Sample keys:", list(dataset[0].keys()) if len(dataset) > 0 else "No samples")

      # Run the evaluation - it now returns two values
      evaluation_results, total_tp_iou_sum = evaluate_layout_model(
          dataset=dataset, # Pass the raw dataset
          prompt_map=filtered_prompt_to_label,
          label_map_rev=label_to_prompt_map,
          iou_threshold=IOU_THRESHOLD,
          num_samples=NUM_SAMPLES
      )

      # Calculate and print metrics - pass the total IoU sum
      if evaluation_results:
          final_metrics = calculate_metrics(evaluation_results, total_tp_iou_sum)
      else:
          print("Evaluation did not produce results.")

      print("\nEvaluation Complete.")