# Installation


Installation steps are consistent with Verl. Please refer to the [Verl Documentation](https://verl.readthedocs.io/en/latest/start/install.html).

**Critical Modifications Required:**  
To ensure successful training, you must modify the following files in your environment's transformers package:

1. Modify `modeling_llava.py` at:  
   `envs/verl/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py`

```python
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
-    # _no_split_modules = ["LlavaVisionAttention"]
+    _no_split_modules = ["CLIPEncoderLayer", "LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
```

2. Add `max_position_embeddings` parameter in `configuration_llava.py` at:  
   `envs/verl/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py`

```python
    model_type = "llava"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
+        max_position_embeddings=32768,
        multimodal_projector_bias=True,
        **kwargs,
    ):
```

# Prepare Dataset

Download the nuScence dataset images from the following links:

- [Google Drive](https://drive.google.com/file/d/1DeosPGYeM2gXSChjMODGsQChZyYDmaUz/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/11xvxPzUY5xTIsJQrYFogqg?pwd=mk95)
- [HuggingFace](https://huggingface.co/datasets/OpenDriveLab/DriveLM/blob/main/drivelm_nus_imgs_train.zip)


Organize the data structure as follows:

```txt
verl
├── data/
|   ├── drivelm
|   │   ├── nuscenes/
|   │   │   ├── samples/
```


# DriveRX Training

Download [Align-DS-V](https://huggingface.co/PKU-Alignment/Align-DS-V)


Fill your path in verl/examples/grpo_trainer/run_dsv.sh


Execute the following command:

```bash
bash examples/grpo_trainer/run_dsv.sh
```



