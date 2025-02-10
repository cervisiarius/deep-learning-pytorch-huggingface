model_id=cotroller_DeepSeek-R1-Distill-Qwen-1-5B_3ep
username=cervisiarius

# Train:
python3.11 scripts/run_cotroller_training.py --config receipes/${model_id}.yaml

# Merge weights:
# python3.11 scripts/merge_adapter_weights.py --peft_model_id runs/${model_id} --output_dir runs/${model_id} --save_tokenizer True
python3.11 scripts/merge_adapter_weights.py --peft_model_id runs/${model_id} --push_to_hub True --repository_id ${model_id}_MERGED

# Inference:
# docker run --name tgi --gpus 1 -d -ti -p 8080:80 --shm-size=2GB -v runs/${model_id}:/data/model ghcr.io/huggingface/text-generation-inference:3.0.1 --model-id /data/model --num-shard 1
docker run --name tgi --gpus 1 -d -ti -p 8080:80 --shm-size=2GB -e HF_TOKEN=$(cat ~/.cache/huggingface/token) ghcr.io/huggingface/text-generation-inference:3.0.1 --model-id ${username}/${model_id}_MERGED --num-shard 1
