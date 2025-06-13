import os
import json
import re
import warnings
import datasets
import random
from datasets import Dataset
from collections import defaultdict
from PIL import Image

import argparse


system_prompt = """You are a smart autonomous driving assistant responsible for analyzing and responding to driving scenarios.
You are provided with up to six camera images in the sequence [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT].

Instructions:

Answer Requirements:

- For binary questions (e.g., "Is there a vehicle in front?"), respond with “Yes” or “No”, and include an explanation referencing relevant visual evidence.

- For open-ended questions, provide a concise yet informative answer focusing on relevant objects, their attributes, and spatial or motion-related details.

- Your answer must be based on visual and contextual understanding.

- You MUST first examine the input images and think step-by-step about the reasoning process before giving the final answer.

- The reasoning process MUST be enclosed within <think>...</think> tags.

Key Information for Driving Context:

- Emphasize object attributes (e.g., category, appearance, status) and motion characteristics (e.g., speed, direction, behavior).

- Prioritize details relevant to driving safety and decision-making, such as potential hazards, traffic dynamics, or maneuver planning.

Use all available images and coordinate information to respond accurately to questions related to perception, prediction, planning, or behavior.
"""


def replace_system_prompt(image_paths, prompt=system_prompt) -> str:
    """
    Replaces the specific sentence in the system prompt to reflect only the provided camera images.

    Args:
        prompt (str): The original system prompt containing the sentence to be replaced.
        image_paths (Dict[str, str]): A dictionary of image file paths corresponding to different cameras.

    Returns:
        str: The updated system prompt with the specified sentence adjusted to include only the available cameras.
    """

    camera_order = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT"
    ]


    camera_pattern = r'samples/([^/]+)/'


    extracted_cameras = []
    for path in image_paths:
        if path is not None and isinstance(path, str):
            match = re.search(camera_pattern, path)
            if match:
                camera_name = match.group(1)
                if camera_name in camera_order:
                    extracted_cameras.append(camera_name)
                else:
                    warnings.warn(f"Unrecognized camera name '{camera_name}' in path '{path}'.")
            else:
                warnings.warn(f"Unable to extract camera name from path '{path}'.")
        else:
            warnings.warn(f"Invalid image path: {path}")


    unique_cameras = []
    seen = set()
    for cam in extracted_cameras:
        if cam not in seen:
            unique_cameras.append(cam)
            seen.add(cam)

    ordered_cameras = [cam for cam in camera_order if cam in unique_cameras]

    if not ordered_cameras:
        warnings.warn("No valid camera images provided. Using default prompt.")
        return prompt
    else:
        cameras_str = ", ".join(ordered_cameras)
        if len(ordered_cameras) == 1:
            new_sentence = f"You are provided with a single camera image: [{cameras_str}]."
        else:
            new_sentence = f"You are provided with {len(ordered_cameras)} camera images in the sequence [{cameras_str}]."


    original_sentence_pattern = r"You are provided with up to six camera images in the sequence \[CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT\]\."

    updated_prompt, num_subs = re.subn(original_sentence_pattern, new_sentence, prompt)

    if num_subs == 0:
        print("Warning: Original sentence not found in the prompt. No replacement made.")

    return updated_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/drivelm_data')
    parser.add_argument("--train_json_path", default="data/drivelm/train.json")
    parser.add_argument("--test_json_path", default="data/drivelm/test.json")
    parser.add_argument("--image_dir", default="data/drivelm")
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    train_path = args.train_json_path
    test_path = args.test_json_path
    
    image_dir = args.image_dir


    with open(train_path, 'r') as f:
        train_raw_data = json.load(f)
    
    with open(test_path, 'r') as f:
        test_raw_data = json.load(f)
    
    test_data = test_raw_data
    train_data = train_raw_data


    random.shuffle(train_data)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)


    def process_item(example, idx, split):

        pil_images = [os.path.join(image_dir, img_path) for img_path in example['image']]
        
        question = example['conversations'][0]['value']
        answer = example['conversations'][1]['value'] if len(example['conversations']) > 1 else ""
        
        assert len(pil_images) == 6, f"data{idx} don't have 6 images\n{question}"

        system_prompt_use = replace_system_prompt(pil_images)

        prompt = "<｜begin▁of▁sentence｜><｜User｜>"
        image_placeholders = "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\n"

        prompt = prompt + image_placeholders + system_prompt_use + question + "\n<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>"
        data = {
            "data_source": "drivelm",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": pil_images,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                "id": example['id'],
                "tag": example['tag'],
                "task": example["task"],
                'index': idx,
                'answer': answer,
                "question": question,
            }
        }
        return data


    train_dataset = train_dataset.map(
        lambda ex, idx: process_item(ex, idx, 'train'),
        with_indices=True,
        num_proc=8
    )
    
    test_dataset = test_dataset.map(
        lambda ex, idx: process_item(ex, idx, 'test'),
        with_indices=True,
        num_proc=8
    )


    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))


if __name__ == "__main__":
    main()

