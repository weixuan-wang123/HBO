export HF_HOME=/mnt/workspace/cache
export HF_TOKEN_PATH=/mnt/workspace/cache/hub/token
export VLLM_CACHE_ROOT=/mnt/workspace/cache/vllm
export VLLM_USE_MODELSCOPE=false
export PROJECT_DIR=/mnt/workspace/dynasample-project

# model=Qwen/Qwen2.5-0.5B
model=Qwen/Qwen2.5-7B # e.g. meta-llama/Llama-3.1-8B 
inter_reward_type=gradnorm # "cossim" or "diff"
intra_reward_type=diff_ppl # "cossim" or "diff"
update_steps=200 # interval for updating the scorer network e.g. 200
temperature=1 # initial sampling probability, e.g. 1, inf
use_ema=no # use ema or not "yes" or "no"
ema_alpha=0.9 # beta in equation 11. e.g. 0.9
dataset_name=dynasample_train_scoreby3llms # dataset directory. e.g. mixture_of_skills/datasets
subset_names=arb_deu_eng_spa_hin_rus_swa_zho # the names of subset, delimited by "_". e.g. "general_math_medical_p3"
trainer_type="dynasample"
alignment_dataset_name=dynasample_align
dataset_pct=0.1 # percentage of the dataset to use
output_dir=/mnt/workspace/dynasample-project/checkpoints/$model-debug # directory saving checkpoint

rm -rf $output_dir

bash HBO/jobscripts/dynasample/dynasample.sh \
    $model $inter_reward_type $intra_reward_type $update_steps $temperature \
    $use_ema $ema_alpha $output_dir $dataset_name $subset_names \
    $trainer_type $alignment_dataset_name $dataset_pct



