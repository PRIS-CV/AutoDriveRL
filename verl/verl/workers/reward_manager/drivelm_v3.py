
import json
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict
from multiprocessing import Pool, cpu_count
import copy
import glob
import torch
import numpy as np
from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import requests
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
import argparse



def get_prompt(item):
    PRED = extract_solution(item["answer"])
    GT = item["gt_answer"]
    task = item["task"]
    tag = item["tag"]

    if task == "perception":
        if tag == 0:
            prompt = f"""
Please evaluate the answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct response and well - supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (50 points):
 - Exact Match (50 points): Assign 50 points if the predicted answer is exactly correct.
 - No Match (0 points): Assign 0 points if the predicted answer is incorrect, regardless of explanation quality.

2. Understanding Depth (10 points):
 - Award up to 10 points based on how thoroughly the answer demonstrates understanding of the situation or problem.

3. Context Awareness (15 points):
 - Score up to 15 points for acknowledging relevant contextual factors that influence the perception or understanding of the situation.
 - Guideline: Deduct points if important contextual factors are ignored or inadequately addressed.

4. Logic and Reasoning (15 points):
 - Award up to 15 points if the explanation presents logical and well - supported reasoning.
 - Guideline: Deduct points for illogical or poorly supported reasoning.

5. Clarity of Reasoning (10 points):
 - Award up to 5 points for clear, logically structured reasoning that is easy to understand.
 - Assign up to 5 points for grammatical accuracy and coherent structure.
 - Guideline: Deduct points for vague or confusing explanations that hinder comprehension. Deduct points for grammar or syntax issues that impact clarity or logical flow.

Here is the correct answer or solution: "{GT}"

Here is the predicted answer and explanation (if any): "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (50 points): 
2. Understanding Depth (10 points):
3. Context Awareness (15 points):
4. Logic and Reasoning (15 points):
5. Clarity of Reasoning (10 points): 
Total Score: 

Brief Summary: 
"""
        else:
            prompt = f"""Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the example correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Information Completeness (30 points):
   - Score based on whether the predicted answer fully covers the key details in the example answer (including vehicle type, location, color, and object IDs). Deduct points for missing or incorrect information.

2. Detail Accuracy (30 points):
   - Assess whether the predicted answer accurately describes details like vehicle location and direction, matching the example answer. Points should be deducted for any deviations or ambiguous descriptions.

3. Clarity of Expression (20 points):
   - Evaluate whether the predicted answer is clearly and coherently expressed, allowing for accurate understanding of the information, similar to the example answer. Points should be deducted for confusing or ambiguous wording.

4. Format Compliance (20 points):
   - Check whether the predicted answer follows a similar format to the example answer, including punctuation and data presentation. Points should be deducted for non-compliant or inconsistent formatting.

Here is the correct answer: "{GT}"

Here is the predicted answer to be evaluated: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Information Completeness (30 points): 
2. Detail Accuracy (30 points):
3. Clarity of Expression (20 points):
4. Format Compliance (20 points):
Total Score: 

Brief Summary:
"""

    elif task == "prediction":
        if tag == 0:
            prompt = f"""Please evaluate the predicted answer for the Yes/No question on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and a well-supported explanation. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (40 points):
- Exact Match (40 points): Assign 40 points if the predicted Yes/No answer exactly matches the correct answer.
- No Match (0 points): Assign 0 points if the predicted answer does not match the correct answer, regardless of explanation quality.
        
2. Object Category Identification (15 points):
- Score up to 15 points for accurately identifying the object's category.
- Guideline: Deduct points for any inaccuracies or missing elements in the category identification, particularly if they affect understanding or recognition of the object’s role in the scene.
        
3. Object Visual Appearance (15 points):
- Score up to 15 points for an accurate description of the object's visual appearance (e.g., colors, materials, size, or shape).
- Guideline: Deduct points if any important visual details are missing, incorrect, or overly generalized, especially if they impact the explanation or perception of the object’s function.

4. Object Position and Motion (15 points):
- Score up to 15 points for correctly identifying the object's location, orientation, and motion (if applicable) relative to the ego vehicle.
- Guideline: Deduct points for inaccuracies in spatial information, positioning, or motion. Include deductions if relevant motion or orientation details are omitted or incorrect.

5. Explanation Clarity and Justification (15 points):
- Score up to 15 points for the clarity, logical structure, and justification of the explanation provided.
- Guideline: Deduct points for vague, confusing, or insufficient explanations that fail to justify the Yes/No answer clearly and logically.

Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the correct answer: "{GT}"

Here is the predicted answer and explanation (if any): "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (40 points):
2. Object Category Identification (15 points):
3. Object Visual Appearance (15 points):
4. Object Position and Motion (15 points):
5. Explanation Clarity and Justification (15 points):
Total Score: 

Brief Summary: """
        else:
            prompt = f"""Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Object Identification and Priority Order (20 points):
- Score up to 20 points for accurately identifying the correct objects in the correct priority order (e.g., first, second, third).
- Guideline: Deduct points for any missed, misidentified, or out-of-sequence objects, particularly if this affects the logic of the driving prediction.

2. Object Category and Visual Description Accuracy (20 points):
- Score up to 20 points for accurately describing each object’s category (e.g., traffic sign, vehicle) and relevant visual attributes (e.g., color, type, size) as necessary for the scene.
- Guideline: Deduct points for incorrect or overly generalized descriptions of categories or visual attributes, especially if they impact scene comprehension or recognition.
  
3. State of the Object (15 points):
- Score up to 15 points based on the accuracy of the object’s state (e.g., moving, stationary) and alignment with the correct answer.
- Guideline: Deduct points if the predicted state does not match the correct state or if critical details of the object’s status are omitted.

4. Recommended Action for Ego Vehicle (15 points):
- Score up to 15 points for accurately identifying the action the ego vehicle should take in response to each object (e.g., continue ahead, slow down).
- Guideline: Deduct points for actions that are inappropriate, unclear, or lacking necessary detail, especially if they contradict the driving context.

5. Logical Flow and Reasonableness of Prediction (20 points):
- Score up to 20 points for the logical consistency and reasonableness of the entire response. The answer should reflect a clear, step-by-step rationale.
- Guideline: Deduct points for inconsistencies, contradictions, or illogical reasoning that undermine the reliability of the prediction.

6. Clarity and Grammar (10 points):
- Score up to 10 points for clarity, coherence, and grammatical accuracy. Assign up to 5 points for logical structure and readability, and up to 5 points for grammatical accuracy.
- Guideline: Deduct points for ambiguous explanations, unclear reasoning, or grammatical errors that reduce clarity.

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Object Identification and Priority Order (20 points):
2. Object Category and Visual Description Accuracy (20 points):
3. State of the Object (15 points):
4. Recommended Action for Ego Vehicle (15 points):
5. Logical Flow and Reasonableness of Prediction (20 points):
6. Clarity and Grammar (10 points):
Total Score: 

Brief Summary: 
"""

    elif task == "planning":

        prompt = f"""Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Action Prediction Accuracy (40 points):
   - Score up to 40 points based on the accuracy of the predicted action for the ego vehicle (e.g., keep going, turn left, turn right, accelerate, decelerate) in response to the contextual information.
   - Guideline: Award full points only for exact or highly similar action matches. Deduct points for inaccuracies or actions that do not match the correct answer, especially if they could compromise safety or appropriateness in context.

2. Reasoning and Justification (20 points):
   - Score up to 20 points for a clear and logical explanation of why the action is chosen, ensuring that the reason aligns with safety, environmental factors, or other relevant considerations.
   - Guideline: Deduct points if the reasoning lacks clarity, omits relevant details, or includes contradictions. The explanation should justify the action in a way that is suitable for the scenario provided.

3. Probability or Confidence Level (15 points):
   - Score up to 15 points for accurately predicting the probability or confidence level of the action being safe or feasible (e.g., high probability if no obstacles are present).
   - Guideline: Deduct points if the probability level is missing, implausible, or does not align with the action or reasoning provided.

4. Contextual Awareness and Safety Considerations (15 points):
   - Score up to 15 points for reflecting an awareness of the driving context, including potential obstacles, traffic participants, and safety implications.
   - Guideline: Deduct points for failing to consider contextual factors that may impact the ego vehicle's decision, especially if they could lead to unsafe actions.

5. Conciseness and Clarity (10 points):
   - Assess the clarity and brevity of the answer. Answers should be concise and easy to understand, effectively communicating the intended actions and rationale.
   - Guideline: Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension. Assign 0 points if no explanation is provided.

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Action Prediction Accuracy (40 points):
2. Reasoning and Justification (20 points):
3. Probability or Confidence Level (15 points):
4. Contextual Awareness and Safety Considerations (15 points):
5. Conciseness and Clarity (10 points):
Total Score:

Brief Summary:
"""

    elif task == "behavior":

        prompt = f"""Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (50 points):
- Score up to 50 points based on how accurately the predicted answer captures the core meaning of the correct answer.
- Guideline: Award full points only for semantically equivalent or highly accurate matches. Deduct points for factual errors, omissions, or meaning distortions. Assign 0 points if the predicted answer is completely incorrect or irrelevant.

2. Behavioral Understanding and Detail (15 points):
- Score up to 15 points for accurately capturing the behavior of the ego vehicle (e.g., direction, speed, maneuvers) as conveyed in the correct answer.
- Guideline: Deduct points if key behavioral details are missing or misrepresented (e.g., wrong turning direction, missed speed changes).

3. Reasoning and Justification (15 points):
- Score up to 15 points for a clear and logical explanation supporting the predicted answer. The reasoning should reflect an understanding of why the described behavior is appropriate in the scenario (e.g., traffic flow, obstacles, signage).
- Guideline: Deduct points if the explanation is vague, inaccurate, irrelevant, or contradicts the correct interpretation.

4. Contextual Relevance (10 points):
- Score up to 10 points for incorporating relevant environmental or situational context (e.g., presence of other vehicles, road layout, objects) into the answer or justification.
- Guideline: Deduct points if the prediction ignores important contextual cues that influence the behavior.

5. Clarity and Grammar (10 points):
- Score up to 10 points for linguistic quality, including clarity, coherence, and grammatical accuracy. The answer should be concise and understandable.
- Guideline: Deduct points for confusing phrasing, poor structure, or grammar issues that interfere with comprehension.

Assign 0 points for criteria 2 to 5 if no explanation is provided.

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (50 points):
2. Behavioral Understanding and Detail (15 points):
3. Reasoning and Justification (15 points):
4. Contextual Relevance (10 points):
5. Clarity and Grammar (10 points):
Total Score: 

Brief Summary:
"""
    return prompt


def extract_score(reply):
    """Extracts an integer score from a line like "Total Score: <score>".
    """
    pattern = r"Total Score:\s*(\d{1,3})\b"
    match = re.search(pattern, reply)
    if match:
        score = int(match.group(1))

        if score < 0:
            return 0
        if score > 100:
            return 100
        return score
    else:
        return -99


def extract_solution(input_str):
    try:
        answer = input_str.split('</think>')[1].strip()
        think = input_str.split('</think>')[0].split('<think>')[1]
    except Exception as e:
        answer = input_str
        
    return answer


def call_vllm_chat(messages, url, max_retries=10, model_name="qwen", **gen_kwarg):
    if not messages:
        return {"error": "messages is empty"}
    
    url = f"{url}/chat/completions"
    headers = {'Content-Type': 'application/json'}
    retries = 0

    kwarg_ls = ["temperature", "top_p", "top_k", "max_tokens"]
    format_gen_kwargs = {k: v for k, v in gen_kwarg.items() if k in kwarg_ls}
    
    data_entry = {
        "messages": messages,
        "stream": False,
        "model": model_name,
        **format_gen_kwargs
    }
    
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, json=data_entry, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f'API Error: {e}')
            random_sleep = random.randint(100, 1500)
            time.sleep(random_sleep/1000)
            retries += 1

    print(f'Get max retries {max_retries}, return None.')
    return None
    








def extract_json_from_string(s):
    """从字符串中提取 JSON 格式的内容"""
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return None
    json_str = s[start:end+1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None



def cal_gpt_score(item_list: List[Dict]) -> List[float]:

    worker_ip="10.80.11.221"
    url = f"http://{worker_ip}:8012/v1"


    def process_item(item: Dict) -> float:
        answer = extract_solution(item["answer"])
        gt_answer = item["gt_answer"]
        

        if answer == item["answer"] or is_abnormal_output(item["answer"]):
        # if answer == item["answer"]:
            return {
                "score": 0,
                "reason": "No think tags or repeat so much",
            }


        messages = [
            {"role": "user", "content": get_prompt(item)}
        ]

        json_none_retry = 0
        temperature = 0
        while json_none_retry < 10:
            response = call_vllm_chat(
                messages=messages,
                url=url,
                max_retries=10,
                model_name="qwen",
                temperature=temperature,
                max_tokens=2048,
            )

            response_str = response["choices"][0]["message"]["content"]
            

            score = extract_score(response_str)
            if score != -99:
                return {
                    "score": score,
                    "reason": response_str,
                }
            
            temperature += 0.1
            json_none_retry += 1

        return {
            "score": -99,
            "reason": response_str,
        }

    json_list = [None] * len(item_list)
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_item, item): i for i, item in enumerate(item_list)}
        
        with tqdm(total=len(futures), desc="Processing") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                json_list[idx] = future.result()
                pbar.update(1)

    return json_list





def get_next_val_filename(output_dir="val_logs"):
    """Get the next val filename with incrementing number"""
    os.makedirs(output_dir, exist_ok=True)
    existing_files = glob.glob(os.path.join(output_dir, "val_*.jsonl"))
    if not existing_files:
        return os.path.join(output_dir, "val_1.jsonl")
    
    max_num = 0
    for f in existing_files:
        try:
            num = int(os.path.basename(f).split('_')[1].split('.')[0])
            if num > max_num:
                max_num = num
        except:
            continue
    
    return os.path.join(output_dir, f"val_{max_num + 1}.jsonl")

def save_val_data(data_items, output_dir="val_logs"):
    """Save validation data to JSONL file"""
    filename = get_next_val_filename(output_dir)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved validation data to {filename}")


class DriveLMV3Manager:
    def __init__(self, tokenizer, num_examine, compute_score=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def _constract_item(self, data: DataProto):
        pass

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))
        tag_list = [info["tag"] for info in extra_info]
        task_list = [info["task"] for info in extra_info]
        prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        item_list = [{
            "prompt": prompt_str[i] if i < len(prompt_str) else "",
            "answer": response_str[i],
            "gt_answer": ground_truth[i],
            "tag": tag_list[i],
            "task": task_list[i]
        } for i in range(len(response_str))]

        json_list = cal_gpt_score(item_list)

        score_list = []
        for i in range(len(json_list)):
            if json_list[i] is not None and json_list[i].get("score") is not None:
                score = json_list[i]["score"]
                if score <= 0:
                    score = 0
                elif score > 100:
                    score = 100
                else:
                    score = score
                
                score /= 100
            else:
                score = None

            score_list.append(score)

        valid_scores = [s for s in score_list if s is not None]
        if valid_scores:
            mean_score = sum(valid_scores) / len(valid_scores)
        else:
            mean_score = 0.0
        
        score_list = [s if s is not None else mean_score for s in score_list]

        for i in range(len(item_list)):
            item_list[i]["score"] = score_list[i]
            item_list[i]["gpt_response"] = json_list[i]

        save_val_data(item_list)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = score_list[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[response]", response_str[i])

        return reward_tensor







