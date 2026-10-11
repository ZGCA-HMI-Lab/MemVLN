#!/usr/bin/bash
set -euo pipefail

# 结果文件默认写入 evaluation/ 下，并使用相对路径组织运行资源。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DATA_ROOT="${DATA_ROOT:-data}"
export WANDB_PROJECT="${WANDB_PROJECT:-MemoryVLN}"

export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

CHECKPOINT="${CHECKPOINT_PATH:-checkpoints/memoryvln}"
# Transformers 需要本地模型目录的明确路径；将相对 checkpoint 路径解析到仓库根目录。
if [[ "${CHECKPOINT}" != /* ]]; then
  CHECKPOINT="${REPO_ROOT}/${CHECKPOINT}"
fi
PROCESSOR_PATH="${PROCESSOR_PATH:-models/Qwen2.5-VL-7B-Instruct}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
DATASET_SUFFIX=$(basename "$CONFIG" .yaml | sed 's/^vln_//')
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/${CHECKPOINT_NAME}_${DATASET_SUFFIX}}"

if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

_USER_WORLD_SIZE="${WORLD_SIZE:-}"
if [[ -n "${_USER_WORLD_SIZE}" ]]; then
  NNODES=1
  NODE_RANK=0
  NPROC_PER_NODE="${_USER_WORLD_SIZE}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
else
  NNODES="${NNODES:-${SLURM_NNODES:-1}}"
  NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
  NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}}"
  if [[ -z "${NPROC_PER_NODE}" ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
      NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
    else
      NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
  fi
  if [[ -z "${NPROC_PER_NODE}" ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    NPROC_PER_NODE=1
  fi
  if [[ "${NNODES}" -gt 1 ]]; then
    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" 2>/dev/null | head -1)}"
    MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
  else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  fi
  MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
fi

TOTAL_EVAL_PROCS=$((NNODES * NPROC_PER_NODE))
RESULT_JSON="${RESULT_JSON:-result_${EVAL_SPLIT}.json}"

# 历史帧抽帧（与 build_vlnce_relative_pose_dataset.py 默认对齐）
_DEFAULT_MAX_IMAGE_FRAMES=-1
MAX_IMAGE_FRAMES="${EVAL_MAX_IMAGE_FRAMES:-$_DEFAULT_MAX_IMAGE_FRAMES}"
EBB_CHUNK="${EVAL_EBBINGHAUS_CHUNK_SIZE:-8}"
EBB_S="${EVAL_EBBINGHAUS_S:-5.0}"
EBB_S_OUTER="${EVAL_EBBINGHAUS_S_OUTER:-3.0}"
SHRINK_START_FRAMES="${EVAL_SHRINK_START_FRAMES:-16}"
SHRINK_ADAPTIVE_OFFSET="${EVAL_SHRINK_ADAPTIVE_OFFSET:-4}"
SHRINK_ADAPTIVE_COEF="${EVAL_SHRINK_ADAPTIVE_COEF:-2}"
SHRINK_MIN_FRAMES="${EVAL_SHRINK_MIN_FRAMES:-16}"
MAX_STEPS="${EVAL_MAX_STEPS:-400}"
EVAL_LIMIT_EPISODES="${EVAL_LIMIT_EPISODES:-0}"
EVAL_PROFILE_INFERENCE_SPEED="${EVAL_PROFILE_INFERENCE_SPEED:-0}"
SEED="${EVAL_SEED:-42}"
EVAL_PORT_VAL="${EVAL_PORT:-1111}"
EVAL_DIST_URL_VAL="${EVAL_DIST_URL:-env://}"
EVAL_DEVICE_VAL="${EVAL_DEVICE:-cuda}"

# 位置编码：默认开启；关闭时传 --no_position_encoding。例：USE_POSITION_ENCODING=0 bash scripts/evaluation.sh
USE_POSITION_ENCODING="${USE_POSITION_ENCODING:-1}"
MAX_MEMORY_FRAMES="${EVAL_MAX_MEMORY_FRAMES:-16}"
MEMORY_SHORT_CAPACITY="${EVAL_MEMORY_SHORT_CAPACITY:-4}"
MEMORY_LONG_CAPACITY="${EVAL_MEMORY_LONG_CAPACITY:-12}"

echo "[evaluation] CHECKPOINT: ${CHECKPOINT}"
echo "[evaluation] PROCESSOR_PATH: ${PROCESSOR_PATH}"
EVAL_HABITAT_DATA_PATH="${EVAL_HABITAT_DATA_PATH:-}"
echo "[evaluation] CONFIG: ${CONFIG} (dataset: ${DATASET_SUFFIX})"
if [[ -n "${EVAL_HABITAT_DATA_PATH}" ]]; then
  echo "[evaluation] EVAL_HABITAT_DATA_PATH: ${EVAL_HABITAT_DATA_PATH}"
else
  echo "[evaluation] EVAL_HABITAT_DATA_PATH: unset（沿用 yaml；RxR 英语可加 .../rxr/{split}/{split}_guide_en.json.gz）"
fi
echo "[evaluation] OUTPUT_PATH: ${OUTPUT_PATH}"
echo "[evaluation] EVAL_SPLIT: ${EVAL_SPLIT}"
echo "[evaluation] RESULT_JSON: ${RESULT_JSON}"
echo "[evaluation] EVAL_LIMIT_EPISODES: ${EVAL_LIMIT_EPISODES} (0=all)"
echo "[evaluation] EVAL_PROFILE_INFERENCE_SPEED: ${EVAL_PROFILE_INFERENCE_SPEED}"
echo "[evaluation] distributed: NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE} total_procs=${TOTAL_EVAL_PROCS}"
echo "[evaluation] USE_POSITION_ENCODING: ${USE_POSITION_ENCODING} (0/false/no/off → disable pose encoding)"
# 显式记忆库：由 checkpoint 内 config.json 的 use_memory_bank 决定；为 true 时 evaluation.py 会传 chunk_meta 并可选调节：
#   EVAL_MEMORY_POOL_FRAMES=32  EVAL_INFERENCE_COMPRESS_MAX_FRAMES=56
echo "[evaluation] memory_bank: 默认读 checkpoint config.json；可传参覆盖：--use_memory_bank | --no_use_memory_bank"
echo "[evaluation] scene_bank: 默认读 config 的 use_scene_memory_bank；可传 EVAL_EXTRA_ARGS：--use_scene_memory_bank | --no_use_scene_memory_bank（需 MemBank）"
echo "[evaluation] frame_confidence: 默认读 config 的 use_frame_confidence；可传：--use_frame_confidence | --no_use_frame_confidence"
echo "[evaluation] memory_compression: 默认读 config 的 use_memory_compression；可传参：--use_memory_compression | --no_memory_compression；可选 EVAL_MEMORY_POOL_FRAMES / EVAL_INFERENCE_COMPRESS_MAX_FRAMES"
# 分阶段评测（默认关）：< N 帧 VGGT 跨步 KV + 8+48 trim；≥ N 帧启用帧级 Memory、每步清空 VGGT KV
EVAL_PHASED_MEMORY="${EVAL_PHASED_MEMORY:-0}"
EVAL_MEMORY_START_FRAMES="${EVAL_MEMORY_START_FRAMES:-32}"
_lower_phased_echo=$(printf '%s' "${EVAL_PHASED_MEMORY}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_phased_echo}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[evaluation] phased_memory: on start_frames=${EVAL_MEMORY_START_FRAMES}（<N: VGGT KV+trim; ≥N: frame Memory+VGGT reset/step）"
else
  echo "[evaluation] phased_memory: off（EVAL_PHASED_MEMORY=1 开启；EVAL_MEMORY_START_FRAMES 默认 32）"
fi
unset _lower_phased_echo
echo "[evaluation] frame_sampling: skip=${EVAL_SKIP_FRAME_SAMPLING:-0} train_aligned=1 start>${SHRINK_START_FRAMES} K=max(min(N,${SHRINK_ADAPTIVE_OFFSET}+${SHRINK_ADAPTIVE_COEF}*sqrt(N)),${SHRINK_MIN_FRAMES}) max_image_frames=${MAX_IMAGE_FRAMES}"
echo "[evaluation] vit_budget: EVAL_SPLIT_VIT_VGGT=${EVAL_SPLIT_VIT_VGGT:-0}（VGGT trim 开启且未显式设置时脚本会默认置 1）；VIT_FRAME_CAP=${VIT_FRAME_CAP:-0}；入口 scripts/evaluation_vit_budget.sh"
# VGGT KV：默认不截断；EVAL_VGGT_KV_TRIM_8_48=1 启用经典 8+48；EVAL_NO_VGGT_KV_TRIM=1 强制关闭（覆盖前者）。
# trim 实际生效且未设置 EVAL_SPLIT_VIT_VGGT 时，脚本默认 EVAL_SPLIT_VIT_VGGT=1；显式 0 可关闭分流。
EVAL_VGGT_KV_TRIM_8_48="${EVAL_VGGT_KV_TRIM_8_48:-0}"
EVAL_NO_VGGT_KV_TRIM="${EVAL_NO_VGGT_KV_TRIM:-0}"
_lower_v8=$(printf '%s' "${EVAL_VGGT_KV_TRIM_8_48}" | tr '[:upper:]' '[:lower:]')
_lower_kv_off=$(printf '%s' "${EVAL_NO_VGGT_KV_TRIM}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_kv_off}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[evaluation] vggt_kv: trim off（EVAL_NO_VGGT_KV_TRIM）"
elif [[ "${_lower_v8}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[evaluation] vggt_kv: trim 8+48（EVAL_VGGT_KV_TRIM_8_48=1）"
else
  echo "[evaluation] vggt_kv: no trim（默认；EVAL_VGGT_KV_TRIM_8_48=1 开启 8+48）"
fi
unset _lower_v8 _lower_kv_off

# 并排可视化：左 RGB，右顶视占据栅格（与 Habitat TopDownMap + observations_to_image 一致）
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-0}"
EVAL_SAVE_VIDEO_RATIO="${EVAL_SAVE_VIDEO_RATIO:-}"
_lower_sv=$(printf '%s' "${EVAL_SAVE_VIDEO}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sv}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[evaluation] save_video: on (RGB + top-down map); ratio=${EVAL_SAVE_VIDEO_RATIO:-1.0}"
else
  echo "[evaluation] save_video: off（设 EVAL_SAVE_VIDEO=1 开启；可选 EVAL_SAVE_VIDEO_RATIO=0.1）"
fi
unset _lower_sv

# ---------- 依赖检查（评测脚本不修改系统或 Python 环境） ----------
_missing_eval_deps=0
if ! python -c "import habitat, habitat_sim, habitat_baselines, quaternion" >/dev/null 2>&1; then
  echo "[error] 缺少 Habitat-Sim、Habitat-Lab、Habitat-Baselines 或 numpy-quaternion。" >&2
  _missing_eval_deps=1
fi
if ! python -c "import ctypes; ctypes.CDLL('libEGL.so.1')" >/dev/null 2>&1; then
  echo "[error] 未找到 libEGL.so.1；请先安装 README 中列出的 Mesa/EGL 系统依赖。" >&2
  _missing_eval_deps=1
fi
if [[ "${_missing_eval_deps}" -ne 0 ]]; then
  echo "[error] 依赖检查失败。请按 README 的 Environment setup 完成安装后重新运行。" >&2
  exit 1
fi
unset _missing_eval_deps

# ffmpeg 仅在保存视频时需要；这里只检查，不自动安装。
_lower_video_dep=$(printf '%s' "${EVAL_SAVE_VIDEO:-0}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_video_dep}" =~ ^(1|true|yes|on)$ ]]; then
  _EVAL_FFMPEG_EXE="${IMAGEIO_FFMPEG_EXE:-}"
  if [[ -z "${_EVAL_FFMPEG_EXE}" ]] && command -v ffmpeg >/dev/null 2>&1; then
    _EVAL_FFMPEG_EXE="$(command -v ffmpeg)"
  fi
  if [[ -z "${_EVAL_FFMPEG_EXE}" ]]; then
    _EVAL_FFMPEG_EXE="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  fi
  if [[ -n "${_EVAL_FFMPEG_EXE}" ]] && [[ -x "${_EVAL_FFMPEG_EXE}" ]]; then
    export IMAGEIO_FFMPEG_EXE="${_EVAL_FFMPEG_EXE}"
    echo "[evaluation] IMAGEIO_FFMPEG_EXE=${IMAGEIO_FFMPEG_EXE}"
  else
    echo "[error] EVAL_SAVE_VIDEO 已开启，但未找到可执行 ffmpeg。请按 README 安装 ffmpeg 或 imageio-ffmpeg。" >&2
    exit 1
  fi
  unset _EVAL_FFMPEG_EXE
fi
unset _lower_video_dep

mkdir -p "$OUTPUT_PATH"
EVAL_LOG="${OUTPUT_PATH}/eval_${EVAL_SPLIT}.log"

EVAL_ARGS=(
  --model_path "$CHECKPOINT"
  --processor_path "$PROCESSOR_PATH"
  --habitat_config_path "$CONFIG"
  --output_path "$OUTPUT_PATH"
  --eval_split "$EVAL_SPLIT"
  --result_json "$RESULT_JSON"
  --max_image_frames "$MAX_IMAGE_FRAMES"
  --shrink_start_frames "$SHRINK_START_FRAMES"
  --shrink_adaptive_offset "$SHRINK_ADAPTIVE_OFFSET"
  --shrink_adaptive_coef "$SHRINK_ADAPTIVE_COEF"
  --shrink_min_frames "$SHRINK_MIN_FRAMES"
  --ebbinghaus_chunk_size "$EBB_CHUNK"
  --ebbinghaus_s "$EBB_S"
  --ebbinghaus_s_outer "$EBB_S_OUTER"
  --max_memory_frames "$MAX_MEMORY_FRAMES"
  --memory_short_capacity "$MEMORY_SHORT_CAPACITY"
  --memory_long_capacity "$MEMORY_LONG_CAPACITY"
  --max_steps "$MAX_STEPS"
  --seed "$SEED"
  --port "$EVAL_PORT_VAL"
  --dist_url "$EVAL_DIST_URL_VAL"
  --device "$EVAL_DEVICE_VAL"
)

if [[ "${EVAL_LIMIT_EPISODES}" -gt 0 ]]; then
  EVAL_ARGS+=(--eval_limit_episodes "${EVAL_LIMIT_EPISODES}")
fi

_lower_profile_speed=$(printf '%s' "${EVAL_PROFILE_INFERENCE_SPEED}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_profile_speed}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--profile_inference_speed)
fi
unset _lower_profile_speed

_lower_eval_pe=$(printf '%s' "${USE_POSITION_ENCODING}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_eval_pe}" =~ ^(0|false|no|off)$ ]]; then
  EVAL_ARGS+=(--no_position_encoding)
fi
unset _lower_eval_pe

_lower_skip=$(printf '%s' "${EVAL_SKIP_FRAME_SAMPLING:-0}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_skip}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--skip_frame_sampling)
fi
unset _lower_skip

# VGGT KV 滑动窗口；与 split 联动：启用 trim 且未显式关闭时默认开启 ViT/VGGT 分流（VGGT 全历史帧）
_lower_v8=$(printf '%s' "${EVAL_VGGT_KV_TRIM_8_48:-0}" | tr '[:upper:]' '[:lower:]')
_lower_no_vggt_trim=$(printf '%s' "${EVAL_NO_VGGT_KV_TRIM:-0}" | tr '[:upper:]' '[:lower:]')
_trim_effective=0
if [[ "${_lower_v8}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--vggt_kv_trim_8_48)
  if ! [[ "${_lower_no_vggt_trim}" =~ ^(1|true|yes|on)$ ]]; then
    _trim_effective=1
  fi
fi
if [[ "${_lower_no_vggt_trim}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--no_vggt_kv_trim)
fi
unset _lower_v8 _lower_no_vggt_trim

if [[ "${_trim_effective}" -eq 1 ]] && [[ ! -v EVAL_SPLIT_VIT_VGGT ]]; then
  EVAL_SPLIT_VIT_VGGT=1
  echo "[evaluation] VGGT trim 已开启且未设置 EVAL_SPLIT_VIT_VGGT → 默认 EVAL_SPLIT_VIT_VGGT=1（若要关闭请显式 EVAL_SPLIT_VIT_VGGT=0）"
fi
unset _trim_effective

_lower_svv=$(printf '%s' "${EVAL_SPLIT_VIT_VGGT:-0}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_svv}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--split_vit_vggt_frames --vit_frame_cap "${VIT_FRAME_CAP:-0}" --vit_sel_strategy "${VIT_SEL_STRATEGY:-tail}")
  echo "[evaluation] split_vit_vggt: ViT=Ebbinghaus(+cap=${VIT_FRAME_CAP:-0}) VGGT=all_frames strategy=${VIT_SEL_STRATEGY:-tail}"
fi
unset _lower_svv

_lower_sv=$(printf '%s' "${EVAL_SAVE_VIDEO:-0}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_sv}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--save_video)
  if [[ -n "${EVAL_SAVE_VIDEO_RATIO}" ]]; then
    EVAL_ARGS+=(--save_video_ratio "${EVAL_SAVE_VIDEO_RATIO}")
  else
    EVAL_ARGS+=(--save_video_ratio 1.0)
  fi
fi
unset _lower_sv

if [[ -n "${EVAL_HABITAT_DATA_PATH}" ]]; then
  EVAL_ARGS+=(--habitat_data_path "${EVAL_HABITAT_DATA_PATH}")
fi

_lower_phased=$(printf '%s' "${EVAL_PHASED_MEMORY}" | tr '[:upper:]' '[:lower:]')
if [[ "${_lower_phased}" =~ ^(1|true|yes|on)$ ]]; then
  EVAL_ARGS+=(--eval_phased_memory --memory_start_frames "${EVAL_MEMORY_START_FRAMES}")
fi
unset _lower_phased

if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a _extra <<< "${EVAL_EXTRA_ARGS}"
  EVAL_ARGS+=("${_extra[@]}")
fi

TORCHRUN_EXTRA=()
if [[ "${NNODES}" -gt 1 ]]; then
  TORCHRUN_EXTRA=(--nnodes="${NNODES}" --node_rank="${NODE_RANK}")
fi

if [[ "${TOTAL_EVAL_PROCS}" -eq 1 ]]; then
  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
  python src/evaluation.py "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_LOG"
else
  torchrun "${TORCHRUN_EXTRA[@]}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/evaluation.py "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_LOG"
fi

echo "[evaluation] done: ${OUTPUT_PATH}/${RESULT_JSON} (MemoryVLN 评测 JSON 数组)"
