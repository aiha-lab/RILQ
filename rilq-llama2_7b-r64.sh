


GPU="1" 
RANK=64
FP_MODEL="your_path/Llama-2-7b-hf"
STEP=10000
LR=1e-4

WD=0.0
BASE_PATH="your_path/"
export MODEL_BASE_PATH="${BASE_PATH}/models"
export HF_CACHE_DIR="${BASE_PATH}/hf_cache"
export HF_DATASETS_CACHE="${BASE_PATH}/hf_datasets"
export CURL_CA_BUNDLE=""
export CUDA_VISIBLE_DEVICES=${GPU}

CALIB_DATASETS='c4'
CALIB_NUM_SAMPLES='320'
CALIB_SEQLEN='512'

base_model="your_path/llama2-7b-rtn-w2a16g64"

output_dir="your_path/llama2-7b-rtn-w2a16g64-rilq-r64"

python3 main.py \
	--dtype "bf16" \
	--base_model ${FP_MODEL} \
	--q_model ${base_model} \
	--lora_r ${RANK} \
	\
	--calib_dataset ${CALIB_DATASETS} \
	--calib_num_samples ${CALIB_NUM_SAMPLES} \
	--calib_val_ratio 0.2 \
	--calib_max_length ${CALIB_SEQLEN} \
	\
	--approx_total_steps ${STEP} \
	--approx_lr ${LR} \
	--approx_batch_size 1 \
	--approx_eval_steps 25 \
	--approx_early_stop \
	--gradient_accumulation_steps 8 \
	\
	--output_dir ${output_dir} 

