set -x

MODEL_PATHS=(

    "STaR/checkpoints/STaR/STaR-Qwen3-0.6B-SFT-Stage1-Stage2/global_step_190/actor/huggingface"
    
)

N_PASSES=8
MAX_LENGTH=4096
TP_SIZE=4

home_dir='STaR'
TEST_FILE="$home_dir/data/STaR-eval.parquet"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUTPUT_DIR="$home_dir/results/$MODEL_PATH"
    mkdir -p ${OUTPUT_DIR}
    mkdir -p "$home_dir/eval_log/$MODEL_PATH"

    python3 -m verl.trainer.our_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${TP_SIZE} \
        data.path=${TEST_FILE} \
        data.output_path=${OUTPUT_DIR}/STaR-eval-${N_PASSES}.parquet \
        data.n_samples=${N_PASSES} \
        data.batch_size=1024 \
        model.path=${MODEL_PATH} \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.max_num_batched_tokens=20480 \
        rollout.temperature=0.6 \
        rollout.prompt_length=8192 \
        rollout.response_length=${MAX_LENGTH} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=${TP_SIZE} $@ 2>&1 | tee >(split -b 5M -d --additional-suffix=.log - "$home_dir/eval_log/$MODEL_PATH/")
done
