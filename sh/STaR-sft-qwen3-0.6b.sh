#!/usr/bin/env bash
set -x

project_name='STaR'
exp_name='STaR-Qwen3-0.6B-SFT'
home_dir='STaR'
mkdir -p "$home_dir/log/$project_name/$exp_name"
save_path="$home_dir/checkpoints/$project_name/$exp_name"
mkdir -p $save_path

MODEL_PATH="$home_dir/model/qwen3-0.6b"
TRAIN_FILE="$home_dir/data/STaR-tqa-sft-addfinqa.parquet"
TEST_FILE="$home_dir/data/STaR-tqa-sft-addfinqa.parquet"

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=256 \
    data.max_length=20480 \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL_PATH \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.total_epochs=3 \
    trainer.logger=['console','swanlab'] $@ 2>&1 | tee >(split -b 5M -d --additional-suffix=.log - "$home_dir/log/$project_name/$exp_name/")
