#!/bin/bash
# Single GPU debug mode - quickly validate pipeline before scaling to 4 GPUs

set -x
export HYDRA_FULL_ERROR=1
export SGL_DISABLE_MEMORY_SAVER=1
export SGLANG_DISABLE_MEMORY_SAVER=1

export SILICONFLOW_API_KEY="sk-iztmlrhucomamftbaowufnonkkjvcnezdsrihgcnmlctlbdd"
export CLAIM_SERVER=https://api.siliconflow.cn/v1/chat/completions
export CLAIM_SERVER_PATH=Qwen/Qwen2.5-32B-Instruct
export CHECK_SERVER=https://api.siliconflow.cn/v1/chat/completions
export CHECK_SERVER_PATH=Qwen/Qwen2.5-32B-Instruct

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_FILE="/root/autodl-tmp/DeepCiteFact/data/rl_train_data_filter_grpo.parquet"
ACTOR_MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen3-8B"
SAVE_PATH="/root/autodl-tmp/output/DeepCiteFact-GRPO-debug"
TOOL_CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/custom_tool_config.yaml"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export DIST_INIT_METHOD="tcp://$MASTER_ADDR:$MASTER_PORT"

# Single GPU config
export CUDA_VISIBLE_DEVICES=0

PROJECT_NAME="DeepCiteFact-GRPO-Debug"
EXPERIMENT_NAME="qwen3-8b-single-gpu-debug"

# Minimal training run: 1 epoch, small batch, focus on startup
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_grpo' \
    reward_model.enable=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TRAIN_FILE \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.format=custom \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=128 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward_model.reward_manager=custom \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.save_freq=999 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode="disable" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=256 $@ 2>&1 | tee grpo_log_debug.txt
