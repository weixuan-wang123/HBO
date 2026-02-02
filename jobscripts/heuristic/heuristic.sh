
ds_config=HBO/deepspeed_config_bf16_stage2.json
run_script=HBO/run_clm.py

model=$1
temperature=$2
subset_names=$3
data_pct=$4
output_dir=$5

# dataset_name=dynasample_train
dataset_name=dynasample_train_scoreby3llms

echo "========================================"
echo "Running $model at temperature $temperature on $subset_names"
echo "Output directory: $output_dir"
echo "========================================"


export WANDB_API_KEY=ed01ba8f6faa724e5e490c56d9dd46b68cc47f04
export WANDB_PROJECT=dynasample
export WANDB_DIR=$output_dir
export WANDB_MODE=online



rm -rf $output_dir
# rm -rf $output_dir/wandb
mkdir -p $output_dir


deepspeed --num_nodes 1 --num_gpus 8 \
    $run_script \
    --deepspeed $ds_config \
    --model_name_or_path $model \
    --dataset_name $dataset_name \
    --subset_names $subset_names \
    --dataset_pct $data_pct \
    --dataset_init_temperature $temperature \
    --trainer_type "base" \
    --output_dir $output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --seed 42 \
    --do_train \
    --bf16 \
    --report_to "wandb" \
    --run_name $(basename $output_dir) \
    --gradient_checkpointing | tee -a /mnt/workspace/logs/$(basename $output_dir).log
