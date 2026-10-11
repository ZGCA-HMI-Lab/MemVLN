#!/usr/bin/env bash
# ViT / VGGT 分流评测：
#   - ViT（对话 + pixel_values）：与训练一致，使用全序列两级艾宾浩斯抽帧。
#   - VGGT：当前仿真步为止的全部 RGB 帧（list(range(total_frames))），不在 VGGT 侧再做艾宾浩斯。
#
# 可选二次省显存：在艾宾浩斯结果上再限制 ViT 帧数，例如 VIT_FRAME_CAP=8。
#
#   EVAL_SPLIT_VIT_VGGT=0 bash scripts/evaluation_vit_budget.sh   # 与 evaluation.sh 完全一致（ViT/VGGT 同源抽帧）
#   VIT_FRAME_CAP=8 bash scripts/evaluation_vit_budget.sh        # ViT 在艾宾浩斯后再最多保留 8 帧
#
set -euo pipefail

# VGGT KV trim 8+48 时 evaluation.sh 会在未设置 EVAL_SPLIT_VIT_VGGT 的情况下默认打开 split。
# 本入口仍以 split 为主：未设置时默认开启分流。
export EVAL_SPLIT_VIT_VGGT="${EVAL_SPLIT_VIT_VGGT:-1}"
export VIT_FRAME_CAP="${VIT_FRAME_CAP:-0}"
export VIT_SEL_STRATEGY="${VIT_SEL_STRATEGY:-tail}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/evaluation.sh" "$@"
