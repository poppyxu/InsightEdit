
import re
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import argparse
import time

def single_mask_to_rle(mask):
    try:
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except Exception as e:
        print(f"Error in single_mask_to_rle: {e}")
        return {}

def find_matching_key_value(word, dictionary):
    def clean_string(s):
        return re.sub(r'[^A-Za-z0-9]', '', s)
    
    cleaned_word = clean_string(word)
    
    for key, value in dictionary.items():
        cleaned_key = clean_string(key)
        if cleaned_key in cleaned_word or cleaned_word in cleaned_key:
            return key, value
    
    return None, None

def process_images(OUTPUT_DIR, images_folder, json_folder, range_index):
    # Create output directory
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        raise
    # Environment settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception as e:
        print(f"Error setting up environment: {e}")
        raise
    # Build SAM2 image predictor
    try:
        SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
        SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
        sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        raise
    
    # Build grounding dino from huggingface
    try:
        GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)
    except Exception as e:
        print(f"Error loading grounding dino model: {e}")
        raise

    # Get image file names
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Calculate range based on the range_index
    start_index = range_index * 10000
    end_index = start_index + 10000

    # Ensure end_index does not exceed the total number of files
    if end_index > len(image_files):
        end_index = len(image_files)
    
    image_files = image_files[start_index:end_index]

    if not image_files:
        print("No images found in the specified range.")
        return

    count = 0

    t1 = time.time()
    # Process each image
    for image_file in image_files:
        try:
            image_name = os.path.splitext(image_file)[0]
            json_file = image_name + '.json'
            json_path = os.path.join(json_folder, json_file)
            
            print(f"Image file: {image_file}")
            
            output_json_path = os.path.join(OUTPUT_DIR, image_file[:-4] + ".json")
            if os.path.exists(output_json_path):
                print(f"文件已存在，跳过保存: {output_json_path}")
                continue
            
            print(count)
            count += 1
            if(count % 100 == 0):
                t2 = time.time()
                print(f"Time taken for 100 images: {t2-t1}")
                t1 = t2

            
            if os.path.exists(json_path):
                print(f"Corresponding JSON file: {json_file}")
            else:
                print("No corresponding JSON file found.")
                continue

            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")
                continue

            caption_str = ""
            detailed_captions = {}

            img_caption = json_data.get("img_caption", "")

            object_list = json_data.get("object_list", [])
            for obj in object_list:
                try:
                    caption = obj["obj_simple_caption"]
                    detailed_caption = obj["obj_detailed_caption"]
                    caption_str += caption + "."
                    detailed_captions[caption.lower()] = detailed_caption
                    
                except Exception as e:
                    print(f"Error processing object in JSON data: {e}")

            img_path = os.path.join(images_folder, image_file)
            text = caption_str
            if text == "":
                continue

            try:
                image = Image.open(img_path)
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue
            try:
                #i1 = np.array(image.convert("RGB"))
                sam2_predictor.set_image(image)
                inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = grounding_model(**inputs)

            except Exception as e:
                print(f"Error processing image with Grounding DINO: {e}")
                continue

            try:
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.size[::-1]]
                )
            except Exception as e:
                print(f"Error in post-processing results: {e}")
                continue

            input_boxes = results[0]["boxes"].cpu().numpy()
            if len(input_boxes) == 0:
                continue

            try:
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            except Exception as e:
                print(f"Error getting predictions from SAM2: {e}")
                continue

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence
                in zip(class_names, confidences)
            ]

            img = cv2.imread(img_path)
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids
            )

            if True:  # DUMP_JSON_RESULTS
                try:


                    mask_rles = [single_mask_to_rle(mask) for mask in masks]
                    input_boxes = input_boxes.tolist()
                    scores = scores.tolist()

                    results = {
                        "image_path": img_path,
                        "img_caption": img_caption,
                        "annotations": [
                            {
                                "class_name": find_matching_key_value(class_name, detailed_captions)[0],
                                "detailed_caption": find_matching_key_value(class_name, detailed_captions)[1],
                                "bbox": box,
                                "segmentation": mask_rle,
                                "score": score,
                            }
                            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
                        ],
                        "box_format": "xyxy",
                        "img_width": image.width,
                        "img_height": image.height,
                    }
                    print("class_name:", results["annotations"][0]["class_name"])
                    #print("detailed_caption:", results["annotations"][0]["detailed_caption"])
                    with open(output_json_path, "w") as f:
                        json.dump(results, f, indent=4)
                    print(f"Saved: {output_json_path}")
                except Exception as e:
                    print(f"Error saving JSON file: {e}")
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

        print("Processing completed.")



def main():
    parser = argparse.ArgumentParser(description="Process images and JSON files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON files.")
    parser.add_argument("--images_folder", type=str, required=True, help="Folder containing input images.")
    parser.add_argument("--json_folder", type=str, required=True, help="Folder containing input JSON files.")
    parser.add_argument("--range", type=int, required=True, help="Range index for processing files.")
    
    args = parser.parse_args()
    
    OUTPUT_DIR = Path(args.output_dir)
    images_folder = args.images_folder
    json_folder = args.json_folder
    range_index = args.range
    
    process_images(OUTPUT_DIR, images_folder, json_folder, range_index)

if __name__ == "__main__":
    main()
