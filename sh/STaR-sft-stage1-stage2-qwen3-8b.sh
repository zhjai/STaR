#!/usr/bin/env bash
set -x

project_name='STaR'
exp_name='STaR-Qwen3-8B-SFT-Stage1-Stage2'
home_dir='STaR'
mkdir -p "$home_dir/log/$project_name/$exp_name"
save_path="$home_dir/checkpoints/$project_name/$exp_name"
mkdir -p $save_path

MODEL_PATH="$home_dir/checkpoints/STaR/STaR-Qwen3-8B-SFT-Stage1/global_step_20/actor/huggingface"
TRAIN_FILE="$home_dir/data/final/STaR-train-hard.parquet"
TEST_FILE="$home_dir/data/final/STaR-test-addfinqa.parquet"

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 4))

train_bsz=256
mini_bsz=256

rollout_n=8
train_temperature=1
val_temperature=0.6

clip_ratio_low=0.2
clip_ratio_high=0.28

ppo_max_token_len_per_gpu=$((1024 * 16))
log_prob_max_token_len_per_gpu=$((1024 * 32))
max_num_batched_tokens=$((1024 * 32))

python3 -m verl.trainer.main_our_grpo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=accurate_score \
    algorithm.filter_groups.threshold_persist=0.99 \
    algorithm.filter_groups.threshold_upper=0.8 \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=$((train_bsz * 2)) \
    data.train_batch_size=${train_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${train_temperature} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.dtype='bfloat16' \
    custom_reward_function.path="$home_dir/reward.py" \
    trainer.default_local_dir=${save_path} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=20 \
    trainer.logger=['console','swanlab'] \
     $@ 2>&1 | tee >(split -b 5M -d --additional-suffix=.log - "$home_dir/log/$project_name/$exp_name/")
