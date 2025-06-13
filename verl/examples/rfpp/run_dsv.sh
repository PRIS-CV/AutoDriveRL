set -x

cp /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/backup/modeling_llava.py $CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py
cp /lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/backup/configuration_llava.py $CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py


export TOKENIZERS_PARALLELISM=true

export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_MODE='offline'

HOME=/lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl

MODEL_PATH=/lpai/volumes/base-rlhf-ali-sh/yanglele/model_zoo/Align-DS-V

PROJECT_NAME=grpo_dsv_run_lix
EXPERIMENT_NAME=bs128_tp2_n8_mb_1_gmu0.6_4k_kl001_rfpp

CKPT_SAVE_PATH=/lpai/volumes/base-rlhf-ali-sh/yanglele/code/verl_ad/verl/examples/rfpp


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$HOME/data/drivelm/train.parquet \
    data.val_files=$HOME/data/drivelm/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=28768 \
    data.max_response_length=4000 \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=drivelm_think \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.default_local_dir=$CKPT_SAVE_PATH/$project_name/$experiment_name \
    trainer.total_epochs=3



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=$HOME/data/drivelm/train.parquet \
    data.val_files=$HOME/data/drivelm/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=28768 \
    data.max_response_length=4000 \
    data.image_key=images \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@



ray stop
