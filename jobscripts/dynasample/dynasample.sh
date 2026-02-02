
export HF_HOME=/mnt/workspace/cache
export HF_TOKEN_PATH=/mnt/workspace/cache/hub/token
export VLLM_CACHE_ROOT=/mnt/workspace/cache/vllm
export VLLM_USE_MODELSCOPE=false
export PROJECT_DIR=/mnt/workspace/dynasample-project

model=$1 # e.g. meta-llama/Llama-3.1-8B
inter_reward_type=$2 # "cossim" or "diff"
intra_reward_type=$3 # "cossim" or "diff"
update_steps=$4 # interval for updating the scorer network e.g. 200
temperature=$5 # initial sampling probability, e.g. 1, inf
use_ema=$6 # use ema or not "yes" or "no"
ema_alpha=$7 # beta in equation 11. e.g. 0.9
output_dir=$8 # directory saving checkpoint
dataset_name=$9 # dataset directory. e.g. mixture_of_skills/datasets
subset_names=${10} # the names of subset, delimited by "_". e.g. "general_math_medical_p3"
trainer_type=${11} # trainer type, e.g. "mos", "dynasample"
alignment_dataset_name=${12} # alignment dataset
dataset_pct=${13} # dataset percentage
ds_config=HBO/deepspeed_config_bf16_stage2.json
run_script=HBO/run_clm.py



# rm -rf $output_dir
mkdir -p $output_dir

deepspeed --num_nodes 1 --num_gpus 8 \
    $run_script \
    --deepspeed $ds_config \
    --model_name_or_path $model \
    --dataset_name $dataset_name \
    --subset_names $subset_names \
    --dataset_pct $dataset_pct \
    --dataset_init_temperature $temperature \
    --trainer_type $trainer_type \
    --sampling_prob_update_steps $update_steps \
    --inter_reward_type $inter_reward_type \
    --intra_reward_type $intra_reward_type \
    --use_ema $use_ema \
    --ema_alpha $ema_alpha \
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
    --gradient_checkpointing \