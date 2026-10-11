#!/bin/bash

# 始终在 MemoryVLN 项目根目录执行（相对路径 src/、scripts/、数据集 json/tsv）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 支持 SLURM 和本机单机/多卡环境。
# SLURM：SLURM_NNODES、SLURM_NODEID、SLURM_NODELIST、SLURM_GPUS_PER_NODE
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}}
if [ -z "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi
[ -z "$NPROC_PER_NODE" ] && NPROC_PER_NODE=8

if [ "$NNODES" -gt 1 ]; then
    MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_NODELIST 2>/dev/null | head -1)}
    MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
else
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}

# 模型默认放在仓库根目录的 models/ 下，可通过环境变量覆盖。
MODEL_ROOT="${MODEL_ROOT:-models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-${MODEL_ROOT}/VGGT-1B}"
# 最终训练数据集：标准 R2R-CE + RxR-CE 相对位姿 TSV。
DATASETS="${DATASETS:-train_r2r_rxr_relative_pose}"

# 位置编码开关：与 train_qwen --use_position_encoding 一致；启用时输出目录名追加 _with_pose（如 qwen2.5vl-7b_32gpu_with_pose）
# 关闭示例：USE_POSITION_ENCODING=0 bash scripts/train.sh
USE_POSITION_ENCODING="${USE_POSITION_ENCODING:-1}"
_lower_pe=$(printf '%s' "${USE_POSITION_ENCODING}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_pe}" =~ ^(0|false|no|off)$ ]]; then
    TRAIN_USE_POSITION_ENCODING="False"
    OUTPUT_POSE_SUFFIX=""
else
    TRAIN_USE_POSITION_ENCODING="True"
    OUTPUT_POSE_SUFFIX="_with_pose"
fi
unset _lower_pe

# 显式记忆库（FusedTokenMemory）：与 train_qwen --use_memory_bank / --lam_mem 一致；关闭示例：USE_MEMORY_BANK=0 bash scripts/train.sh
USE_MEMORY_BANK="${USE_MEMORY_BANK:-0}"
# 未设置时不传 --lam_mem，由 checkpoint config + safetensors 中的 lam_mem 权重继承
LAM_MEM="${LAM_MEM:-}"
_lower_mb=$(printf '%s' "${USE_MEMORY_BANK}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_mb}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_USE_MEMORY_BANK="True"
    OUTPUT_MEM_SUFFIX="_membank"
else
    TRAIN_USE_MEMORY_BANK="False"
    OUTPUT_MEM_SUFFIX=""
fi
unset _lower_mb

# 记忆模块内可微帧压缩（compress_to_memory_tokens + 裁剪 image_pad）；关闭示例：USE_MEMORY_COMPRESSION=0（仅融合+lam_mem）
USE_MEMORY_COMPRESSION="${USE_MEMORY_COMPRESSION:-1}"
_lower_mc=$(printf '%s' "${USE_MEMORY_COMPRESSION}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_mc}" =~ ^(0|false|no|off)$ ]]; then
    TRAIN_USE_MEMORY_COMPRESSION="False"
else
    TRAIN_USE_MEMORY_COMPRESSION="True"
fi
unset _lower_mc

# 帧级记忆池容量（与 train_qwen --max_memory_frames / memory_*_capacity 一致）
MAX_MEMORY_FRAMES="${MAX_MEMORY_FRAMES:-16}"
MEMORY_SHORT_CAPACITY="${MEMORY_SHORT_CAPACITY:-4}"
MEMORY_LONG_CAPACITY="${MEMORY_LONG_CAPACITY:-12}"

# 跨场景 Scene Bank：需同时 USE_MEMORY_BANK=1；chunk_meta 由数据侧从 TSV 的 id 字段自动推断
USE_SCENE_MEMORY_BANK="${USE_SCENE_MEMORY_BANK:-0}"
_lower_sb=$(printf '%s' "${USE_SCENE_MEMORY_BANK}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sb}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_USE_SCENE_MEMORY_BANK="True"
    OUTPUT_SCENE_SUFFIX="_scene"
else
    TRAIN_USE_SCENE_MEMORY_BANK="False"
    OUTPUT_SCENE_SUFFIX=""
fi
unset _lower_sb

SCENE_BANK_CAPACITY="${SCENE_BANK_CAPACITY:-300}"
SCENE_RETRIEVAL_THRESHOLD="${SCENE_RETRIEVAL_THRESHOLD:-0.6}"
SCENE_WARM_TAU="${SCENE_WARM_TAU:-0.5}"
SCENE_TOP_K="${SCENE_TOP_K:-1}"
USE_SCENE_INIT_LOSS="${USE_SCENE_INIT_LOSS:-0}"
SCENE_INIT_LOSS_WEIGHT="${SCENE_INIT_LOSS_WEIGHT:-0.05}"
SCENE_BANK_DDP_SYNC="${SCENE_BANK_DDP_SYNC:-0}"
USE_FRAME_CONFIDENCE="${USE_FRAME_CONFIDENCE:-0}"
MEMORY_CONF_TAU="${MEMORY_CONF_TAU:-5.0}"
MEMORY_CONF_MAX="${MEMORY_CONF_MAX:-10.0}"
MEMORY_WRITEBACK_CONF_THRESHOLD="${MEMORY_WRITEBACK_CONF_THRESHOLD:-0.4}"
SCENE_KEY_LANG_WEIGHT="${SCENE_KEY_LANG_WEIGHT:-0.3}"
SCENE_USE_INSTRUCTION_KEY="${SCENE_USE_INSTRUCTION_KEY:-0}"

_lower_sil=$(printf '%s' "${USE_SCENE_INIT_LOSS}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sil}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_USE_SCENE_INIT_LOSS="True"
else
    TRAIN_USE_SCENE_INIT_LOSS="False"
fi
unset _lower_sil

_lower_sddp=$(printf '%s' "${SCENE_BANK_DDP_SYNC}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sddp}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_SCENE_BANK_DDP_SYNC="True"
else
    TRAIN_SCENE_BANK_DDP_SYNC="False"
fi
unset _lower_sddp

_lower_fc=$(printf '%s' "${USE_FRAME_CONFIDENCE}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_fc}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_USE_FRAME_CONFIDENCE="True"
else
    TRAIN_USE_FRAME_CONFIDENCE="False"
fi
unset _lower_fc

_lower_sik=$(printf '%s' "${SCENE_USE_INSTRUCTION_KEY}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sik}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_SCENE_USE_INSTRUCTION_KEY="True"
else
    TRAIN_SCENE_USE_INSTRUCTION_KEY="False"
fi
unset _lower_sik

# Selective LLM fine-tuning: freeze the backbone except the last N decoder layers.
UNFREEZE_LLM_LAST_N_LAYERS="${UNFREEZE_LLM_LAST_N_LAYERS:-0}"
PARTIAL_LLM_LR="${PARTIAL_LLM_LR:-2e-6}"
ACTION_TOKEN_TEXTS="${ACTION_TOKEN_TEXTS:-STOP,MOVE_FORWARD,TURN_LEFT,TURN_RIGHT}"
TUNE_LM_HEAD="${TUNE_LM_HEAD:-0}"
_PARTIAL_REQUESTED_LR="${LEARNING_RATE:-}"
if ! [[ "$UNFREEZE_LLM_LAST_N_LAYERS" =~ ^[0-9]+$ ]]; then
    echo "[error] UNFREEZE_LLM_LAST_N_LAYERS must be a non-negative integer" >&2
    exit 1
fi

# LoRA：USE_LORA=1 时冻结 LLM 基座、只训 LoRA + merger/lam_mem；比全参 LLM 省显存/时间
# 示例：USE_LORA=1 LORA_R=16 bash scripts/train.sh
USE_LORA="${USE_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_TRAIN_LM_HEAD="${LORA_TRAIN_LM_HEAD:-1}"
LORA_MERGE_BEFORE_SAVE="${LORA_MERGE_BEFORE_SAVE:-0}"
TUNE_MM_LLM="${TUNE_MM_LLM:-}"
TUNE_MM_MLP="${TUNE_MM_MLP:-True}"
LEARNING_RATE="${LEARNING_RATE:-}"

_lower_lora=$(printf '%s' "${USE_LORA}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_lora}" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_USE_LORA="True"
    OUTPUT_LORA_SUFFIX="_lora"
    TUNE_MM_LLM="${TUNE_MM_LLM:-False}"
    LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    LORA_ARGS=(
        --use_lora True
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
        --lora_train_lm_head "$([ "${LORA_TRAIN_LM_HEAD}" = "1" ] && echo True || echo False)"
    )
    if [ -n "$LORA_TARGET_MODULES" ]; then
        LORA_ARGS+=(--lora_target_modules "$LORA_TARGET_MODULES")
    fi
    _lower_lms=$(printf '%s' "${LORA_MERGE_BEFORE_SAVE}" | tr '[:upper:]' '[:lower:]')
    if [[ "${_lower_lms}" =~ ^(1|true|yes|on)$ ]]; then
        LORA_ARGS+=(--lora_merge_before_save True)
    fi
    unset _lower_lms
else
    TRAIN_USE_LORA="False"
    OUTPUT_LORA_SUFFIX=""
    LORA_ARGS=()
    TUNE_MM_LLM="${TUNE_MM_LLM:-True}"
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
fi
unset _lower_lora

OUTPUT_PARTIAL_SUFFIX=""
if [ "$UNFREEZE_LLM_LAST_N_LAYERS" -gt 0 ]; then
    if [ "$TRAIN_USE_LORA" = "True" ]; then
        echo "[error] USE_LORA and UNFREEZE_LLM_LAST_N_LAYERS cannot be enabled together" >&2
        exit 1
    fi
    TUNE_MM_LLM="False"
    LEARNING_RATE="${_PARTIAL_REQUESTED_LR:-$PARTIAL_LLM_LR}"
    TRAIN_ACTION_TOKEN_ROWS="${TRAIN_ACTION_TOKEN_ROWS:-1}"
    OUTPUT_PARTIAL_SUFFIX="_last${UNFREEZE_LLM_LAST_N_LAYERS}"
else
    TRAIN_ACTION_TOKEN_ROWS="${TRAIN_ACTION_TOKEN_ROWS:-0}"
fi
unset _PARTIAL_REQUESTED_LR

_lower_action_rows=$(printf "%s" "$TRAIN_ACTION_TOKEN_ROWS" | tr "[:upper:]" "[:lower:]")
if [[ "$_lower_action_rows" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_ACTION_TOKEN_ROWS_BOOL="True"
else
    TRAIN_ACTION_TOKEN_ROWS_BOOL="False"
fi
unset _lower_action_rows

_lower_tune_head=$(printf "%s" "$TUNE_LM_HEAD" | tr "[:upper:]" "[:lower:]")
if [[ "$_lower_tune_head" =~ ^(1|true|yes|on)$ ]]; then
    TRAIN_TUNE_LM_HEAD="True"
else
    TRAIN_TUNE_LM_HEAD="False"
fi
unset _lower_tune_head

if [ "$TRAIN_ACTION_TOKEN_ROWS_BOOL" = "True" ] && [ "$TRAIN_TUNE_LM_HEAD" = "True" ]; then
    echo "[error] TRAIN_ACTION_TOKEN_ROWS and TUNE_LM_HEAD cannot both be enabled" >&2
    exit 1
fi

PARTIAL_FT_ARGS=(
    --unfreeze_llm_last_n_layers "$UNFREEZE_LLM_LAST_N_LAYERS"
    --tune_lm_head "$TRAIN_TUNE_LM_HEAD"
    --train_action_token_rows "$TRAIN_ACTION_TOKEN_ROWS_BOOL"
    --action_token_texts "$ACTION_TOKEN_TEXTS"
)

# 输出目录：显式 OUTPUT_DIR 优先；否则 OUTPUT_BASE_DIR + MODEL_NAME + 总 GPU 数 [+ _with_pose] [+ _membank] [+ _scene] [+ OUTPUT_EXTRA_SUFFIX]
num_gpus=$((NNODES * NPROC_PER_NODE))
model_name="${MODEL_NAME:-qwen2.5vl-7b}"
if [ -n "${OUTPUT_DIR:-}" ]; then
    :
else
    base_dir="${OUTPUT_BASE_DIR:-outputs}"
    OUTPUT_DIR="${base_dir}/${model_name}_${num_gpus}gpu${OUTPUT_MEM_SUFFIX}${OUTPUT_SCENE_SUFFIX}${OUTPUT_LORA_SUFFIX}${OUTPUT_PARTIAL_SUFFIX}${OUTPUT_EXTRA_SUFFIX:-}"
fi

CACHE_DIR="${CACHE_DIR:-./cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./cache/triton}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRITON_CACHE_DIR/autotune"

_lam_mem_log="${LAM_MEM:-<inherit checkpoint>}"
_global_bs=$((num_gpus * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "NNODES=$NNODES NODE_RANK=$NODE_RANK NPROC_PER_NODE=$NPROC_PER_NODE -> $num_gpus GPUs, per_device_batch=$PER_DEVICE_TRAIN_BATCH_SIZE grad_accum=$GRADIENT_ACCUMULATION_STEPS global_batch=$_global_bs, use_position_encoding=$TRAIN_USE_POSITION_ENCODING, use_memory_bank=$TRAIN_USE_MEMORY_BANK lam_mem=${_lam_mem_log} use_memory_compression=$TRAIN_USE_MEMORY_COMPRESSION, use_scene_memory_bank=$TRAIN_USE_SCENE_MEMORY_BANK, use_lora=$TRAIN_USE_LORA unfreeze_last_n=$UNFREEZE_LLM_LAST_N_LAYERS action_token_rows=$TRAIN_ACTION_TOKEN_ROWS_BOOL tune_mm_llm=$TUNE_MM_LLM lr=$LEARNING_RATE, memory_frames=${MAX_MEMORY_FRAMES}(short=${MEMORY_SHORT_CAPACITY}+long=${MEMORY_LONG_CAPACITY}), OUTPUT_DIR=$OUTPUT_DIR"
unset _global_bs
unset _lam_mem_log

LAM_MEM_ARGS=()
if [ -n "${LAM_MEM}" ]; then
    LAM_MEM_ARGS=(--lam_mem "$LAM_MEM")
fi

# 保证当前 MemoryVLN 项目的 qwen_vl / Memory 优先于环境中的其它 pip -e 同名包（train_qwen 仅 append 不够）
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# MemoryVLN 的 NCCL / 训练环境，可按集群配置调整
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-MemoryVLN}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
REPORT_TO="${REPORT_TO:-none}"
RUN_NAME="${RUN_NAME:-MemoryVLN_${model_name}_${num_gpus}gpu${OUTPUT_POSE_SUFFIX}${OUTPUT_MEM_SUFFIX}${OUTPUT_SCENE_SUFFIX}${OUTPUT_LORA_SUFFIX}${OUTPUT_PARTIAL_SUFFIX}${OUTPUT_EXTRA_SUFFIX:-}}"

# 每卡 micro-batch；显存紧时保持 1。提到 2 可减半 optimizer step 数（总样本吞吐相近时略快）。
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

TORCHRUN_EXTRA=""
[ "$NNODES" -gt 1 ] && TORCHRUN_EXTRA="--nnodes=$NNODES --node_rank=$NODE_RANK"

torchrun $TORCHRUN_EXTRA \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    src/qwen_vl/train/train_qwen.py \
    --model_name_or_path "$MODEL_PATH" \
    --vggt_model_path "$VGGT_MODEL_PATH" \
    --tune_mm_llm "$TUNE_MM_LLM" \
    "${PARTIAL_FT_ARGS[@]}" \
    --tune_mm_vision False \
    --tune_mm_mlp "$TUNE_MM_MLP" \
    "${LORA_ARGS[@]}" \
    --dataset_use "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --bf16 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 163840 \
    --data_flatten False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $((1664*28*28)) \
    --video_min_frame_pixels $((256*28*28)) \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --group_by_modality_length true \
    --seed 42 \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    --reference_frame first \
    --use_position_encoding "$TRAIN_USE_POSITION_ENCODING" \
    --use_memory_bank "$TRAIN_USE_MEMORY_BANK" \
    "${LAM_MEM_ARGS[@]}" \
    --use_memory_compression "$TRAIN_USE_MEMORY_COMPRESSION" \
    --max_memory_frames "$MAX_MEMORY_FRAMES" \
    --memory_short_capacity "$MEMORY_SHORT_CAPACITY" \
    --memory_long_capacity "$MEMORY_LONG_CAPACITY" \
    --use_scene_memory_bank "$TRAIN_USE_SCENE_MEMORY_BANK" \
    --scene_bank_capacity "$SCENE_BANK_CAPACITY" \
    --scene_retrieval_threshold "$SCENE_RETRIEVAL_THRESHOLD" \
    --scene_warm_tau "$SCENE_WARM_TAU" \
    --scene_top_k "$SCENE_TOP_K" \
    --use_scene_init_loss "$TRAIN_USE_SCENE_INIT_LOSS" \
    --scene_init_loss_weight "$SCENE_INIT_LOSS_WEIGHT" \
    --scene_bank_ddp_sync "$TRAIN_SCENE_BANK_DDP_SYNC" \
    --use_frame_confidence "$TRAIN_USE_FRAME_CONFIDENCE" \
    --memory_conf_tau "$MEMORY_CONF_TAU" \
    --memory_conf_max "$MEMORY_CONF_MAX" \
    --memory_writeback_conf_threshold "$MEMORY_WRITEBACK_CONF_THRESHOLD" \
    --scene_key_lang_weight "$SCENE_KEY_LANG_WEIGHT" \
    --scene_use_instruction_key "$TRAIN_SCENE_USE_INSTRUCTION_KEY" \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y-%m-%d_%H%M).log"
