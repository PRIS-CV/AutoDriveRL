# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        if image['bytes']:
            image = Image.open(BytesIO(image['bytes']))
        elif image['path']:
            image = Image.open(image['path'])
    elif isinstance(image, str):
        image = Image.open(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


#     "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- for content in message['content'] %}{% set ns.system_prompt = content['text'] %}{%- endfor %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' and message['content'] is not none %}{{'<｜User｜>'}}{%- for content in message['content'] %}{% if content['type'] == 'text' %}{{content['text']}}{% elif content['type'] == 'image' %}{{'<image>'}}{% endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant'%}{{'<｜Assistant｜>'}}{%- for content in message['content'] %}{% if content['type'] == 'text' %}{{content['text']}}{% elif content['type'] == 'image' %}{{'<image>'}}{% endif %}{%- endfor %}{%- endif %}{%- endfor -%}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}",



def convert_chat(chat, add_generation_prompt=True):
    bos_token = "<｜begin▁of▁sentence｜>"
    prompt = ""
    system_prompt = chat[0]["content"]

    prompt += bos_token + system_prompt

    # Step 2: 构造多轮对话
    for message in chat:
        role = message.get("role")
        content_list = message.get("content", [])

        if role == "user" and content_list:
            prompt += "<｜User｜>"
            for content in content_list:
                prompt += content

        elif role == "assistant":
            prompt += "<｜Assistant｜>"
            for content in content_list:
                prompt += content

    # Step 3: 判断是否添加 Assistant 生成提示符
    if add_generation_prompt:
        prompt += "<｜Assistant｜>"

    return prompt


# def convert_chat(chat):
#     import numpy
#     prompt = ""
#     if isinstance(chat, numpy.ndarray):
#         for i, data in enumerate(chat):
#             if data["role"] == "system":
#                 prompt += f"<｜begin▁of▁sentence｜><｜System｜>"+data["content"]+"<｜end▁of▁sentence｜>\n\n"
#             elif data["role"] == "user":
#                 prompt += f"<｜begin▁of▁sentence｜><｜User｜>"+data["content"]+"<｜end▁of▁sentence｜>\n\n"
#             else:
#                 prompt += f"<｜begin▁of▁sentence｜>"+data["content"]+"<｜end▁of▁sentence｜>\n\n"
            
#             if i == (len(chat) - 1):
#                 prompt += "\n<｜begin▁of▁sentence｜><｜Assistant｜>"
    
#     return prompt


class GRPODatasetDsV(RLHFDataset):
    # def __getitem__(self, item):
    #     """
    #     最终的键prompt_ids参与loss计算, raw_prompt_ids不参与, 但参与val和roll的生成, val的生成需要改一下
    #     Note that we also return the raw_input_ids so that it can be combined with other chat template
    #     """
    #     row_dict: dict = self.dataframe.iloc[item].to_dict()

    #     chat = row_dict.pop(self.prompt_key)

    #     # prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
    #     # prompt_with_chat_template = f"<｜begin▁of▁sentence｜><｜User｜>"+chat+"<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>"
    #     prompt_with_chat_template = convert_chat(chat)
    #     raw_prompt = prompt_with_chat_template

    #     is_multi_modal = self.image_key in row_dict
        
    #     if is_multi_modal:
    #         row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
    #         inputs = self.processor(images=row_dict['multi_modal_data']['image'], text=prompt_with_chat_template, return_attention_mask=True, return_tensors='pt')
    #         # prompt_with_chat_template = self.tokenizer.decode(inputs["input_ids"][0])
    #         pixel_values = inputs['pixel_values']
    #         row_dict['multi_modal_inputs'] = {"pixel_values": None}
    #         row_dict['multi_modal_inputs']['pixel_values'] = pixel_values
    #     else:
    #         inputs = self.processor(text=prompt_with_chat_template, return_attention_mask=True, return_tensors='pt')

    #     prompt_with_chat_template = prompt_with_chat_template.replace("<image>", "<image>"*576)
        
    #     input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
    #                                                                      tokenizer=self.tokenizer,
    #                                                                      max_length=self.max_prompt_length,
    #                                                                      pad_token_id=self.tokenizer.pad_token_id,
    #                                                                      left_pad=True,
    #                                                                      truncation=self.truncation)


    #     position_ids = compute_position_id_with_mask(attention_mask)

    #     row_dict['input_ids'] = input_ids[0]
    #     row_dict['attention_mask'] = attention_mask[0]
    #     row_dict['position_ids'] = position_ids[0]
    #     # row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

    #     #####################################################
    #     # build tag special prompt
    #     #####################################################
    #     import copy
    #     task_chat = copy.deepcopy(chat)
    #     task_system_prompt = (
    #         "You are a driving expert. You will receive six real-time images captured simultaneously from six cameras positioned at the front, front-left, front-right, back, back-left, and back-right of the vehicle, along with a question.\n\n"
    #         "Your task is to carefully analyze all images, reason through your observations as if engaging in an internal monologue, and provide a final answer.\n\n"
    #         "Ensure that your analysis is structured and logical, considering all six perspectives before concluding.\n\n"
    #         "Additionally, some questions may contain references to objects in the format as <c1, CAM_BACK, 800.8, 500.8>. These c tags represent key objects in the scene and follow this structure: \n"
    #         "- c1: The unique identifier of the object.\n"
    #         "- CAM_BACK: The camera in which the object's center point is located (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.)\n"
    #         "- 800.8, 500.8: The 2D bounding box center coordinates (x, y) in the respective camera’s coordinate system.\n\n"
    #         "You FIRST look input images and think about the reasoning process and then provide the final answer.\n"
    #         "The reasoning process MUST BE enclosed within <think> </think> tags.\n"
    #     )
    #     tag = row_dict.get("extra_info", {}).get("tag")
    #     special_instruction = ""
    #     if tag == 2:
    #         special_instruction = (
    #             "Your answer should first identify the objects of interest and then list their coordinates. "
    #             "For example: There is a white sedan to the front of the ego vehicle, a black commercial vehicle to the back of the ego vehicle, and a yellow car to the front left of the ego vehicle. The IDs of these objects are <c1,CAM_FRONT,1011.7,463.3>, <c2,CAM_BACK,676.7,423.3>, and <c3,CAM_FRONT_LEFT,319.2,611.7>."
    #             "You MUST enclose your reasoning process within <think> </think> tags before providing the final answer.\n\n"
    #         )

    #     elif tag == 3:
    #         special_instruction = (
    #             "You MUST answer in sequential order, ensuring that each response follows the specified format: <cx, CAM_xxx, xxx, xxx>.\n\n"
    #             "Response Structure:\n"
    #             "1. Identify and describe the first object the ego vehicle should notice.\n"
    #             "2. Specify the object's state.\n"
    #             "3. Provide an appropriate action for the ego vehicle.\n"
    #             "4. Repeat for the second and third objects.\n\n"
    #             "You MUST enclose your reasoning process within <think> </think> tags before providing the final answer.\n\n"
    #         )
            
    #     task_chat[0]["content"] = task_system_prompt
    #     if special_instruction != "":
    #         task_chat[1]["content"] += " " + special_instruction


    #     task_prompt = convert_chat(task_chat)

    #     row_dict['raw_prompt_ids'] = self.tokenizer.encode(task_prompt, add_special_tokens=True)


    #     # encode prompts without chat template
    #     if self.return_raw_chat:
    #         row_dict['raw_prompt'] = chat.tolist()

    #     # add index for each prompt
    #     index = row_dict.get("extra_info", {}).get("index", 0)
    #     row_dict["index"] = index
        
    #     # if use reward model:
    #     row_dict["reward_model"]["style"] = "model"

    #     return row_dict


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        # prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
        # prompt_with_chat_template = f"<｜begin▁of▁sentence｜><｜User｜>"+chat+"<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜><｜Assistant｜>"
        prompt_with_chat_template = convert_chat(chat)
        raw_prompt = prompt_with_chat_template

        is_multi_modal = self.image_key in row_dict
        
        if is_multi_modal:
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            inputs = self.processor(images=row_dict['multi_modal_data']['image'], text=prompt_with_chat_template, return_attention_mask=True, return_tensors='pt')
            # prompt_with_chat_template = self.tokenizer.decode(inputs["input_ids"][0])
            pixel_values = inputs['pixel_values']
            row_dict['multi_modal_inputs'] = {"pixel_values": None}
            row_dict['multi_modal_inputs']['pixel_values'] = pixel_values
        else:
            inputs = self.processor(text=prompt_with_chat_template, return_attention_mask=True, return_tensors='pt')

        prompt_with_chat_template = prompt_with_chat_template.replace("<image>", "<image>"*576)
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)


        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        
        # if use reward model:
        row_dict["reward_model"]["style"] = "model"

        return row_dict