CUDA_VISIBLE_DEVICES=2 python ./src/export_model.py \
    --model_name_or_path /home/zzj/LLMs/mistral-7b/snapshots/7ad5799710574ba1c1d953eba3077af582f3a773 \
    --adapter_name_or_path ../../saved_model/llm/mistral-7b-lora \
    --template default \
    --finetuning_type lora \
    --export_dir ../../saved_model/llm/mistral-7b-merge \
    --export_size 2 \
    --export_legacy_format False