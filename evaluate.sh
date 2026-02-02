

export HF_HOME=HBO/HBOcache
export MODELSCOPE_CACHE=HBO/HBOcache
export VLLM_CACHE_ROOT=HBO/HBOcache
export VLLM_USE_MODELSCOPE=false
export HF_ENDPOINT=https://hf-mirror.com

model_name_or_path=gpt2

lm_eval --model hf \
    --model_args pretrained=$model_name_or_path,device=cpu,dtype=auto \
    --tasks xcopa \
    --batch_size 1 \
    --limit 10 \
    --output_path /output \