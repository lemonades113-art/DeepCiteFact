# run on 4x A800 80G (SGLang + Multi-turn GRPO)
# Optimized for DeepCiteFact multi-turn tool calling

set -x

export HYDRA_FULL_ERROR=1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

TRAIN_FILE="/root/autodl-tmp/DeepCiteFact/data/rl_train_data_filter_grpo.parquet"
TEST_FILE=$TRAIN_FILE
ACTOR_MODEL_PATH="/root/autodl-tmp/models/Qwen3-8B"

SAVE_PATH="/root/autodl-tmp/output/DeepCiteFact-GRPO"

TOOL_CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/custom_tool_config.yaml"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export DIST_INIT_METHOD="tcp://$MASTER_ADDR:$MASTER_PORT"

PROJECT_NAME="DeepCiteFact-GRPO"
EXPERIMENT_NAME="qwen3-8b-deepcitefact-grpo"


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=64 \
    data.max_prompt_length=1536 \
    data.max_response_length=512 \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style='cosine' \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_cuda_graph=False \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.format=custom \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.allow_auto_truncate=True \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward_model.enable=False \
    reward_model.reward_manager=custom \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode="disable" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@ 2>&1 | tee grpo_log.txt
