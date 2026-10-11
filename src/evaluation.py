import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import re
import glob
import tempfile
import tqdm
import torch
import copy
import json
import random
import argparse
import quaternion
import transformers
import time
import numpy as np
import torch.distributed as dist

from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat.config.default import get_agent_config
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image

from utils.dist import *
import base64
from datetime import datetime
from io import BytesIO
from qwen_vl_utils import extract_vision_info
from transformers import AutoConfig, AutoTokenizer, AutoProcessor
from qwen_vl.model.vggt.utils.load_fn import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForMemoryVLN
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from Memory.memory import two_level_ebbinghaus_indices


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _episode_record_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('scene_id')}|{rec.get('episode_id')}|{rec.get('trajectory_id', '')}"


def _episode_obj_key(ep) -> str:
    tid = getattr(ep, "trajectory_id", None)
    return f"{ep.scene_id}|{ep.episode_id}|{tid}"


def _partial_result_path(result_path: str, rank: int) -> str:
    root, fname = os.path.split(result_path)
    stem, ext = os.path.splitext(fname)
    ext = ext if ext else ".json"
    name = f"{stem}.partial_rank{rank}{ext}"
    return os.path.join(root, name) if root else name


def _all_partial_result_paths(result_path: str) -> List[str]:
    root, fname = os.path.split(result_path)
    stem, ext = os.path.splitext(fname)
    ext = ext if ext else ".json"
    pattern = os.path.join(root or ".", f"{stem}.partial_rank*{ext}")
    return sorted(glob.glob(pattern))


def _atomic_write_json(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp", prefix=".evaljson_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _remove_partial_result_files(result_path: str) -> None:
    for p in _all_partial_result_paths(result_path):
        try:
            os.remove(p)
        except OSError:
            pass


def _metrics_to_jsonable(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k == "top_down_map":
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, (float, int, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = str(v)
    return out


def _write_inference_profile_summary(
    output_path: str,
    records: List[Dict[str, Any]],
    filename: str = "inference_profile_summary.json",
) -> None:
    if not records:
        return

    def _stats(vals: List[float]) -> Dict[str, Any]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean_s": float(arr.mean()),
            "median_s": float(np.median(arr)),
            "p90_s": float(np.percentile(arr, 90)),
            "min_s": float(arr.min()),
            "max_s": float(arr.max()),
            "steps_per_s": float(1.0 / arr.mean()) if float(arr.mean()) > 0 else 0.0,
        }

    times = [float(r["call_model_wall_s"]) for r in records]
    by_step: Dict[str, List[float]] = {}
    by_episode: Dict[str, List[float]] = {}
    for r in records:
        by_step.setdefault(str(r.get("step_id")), []).append(float(r["call_model_wall_s"]))
        ep_key = f"{r.get('scene_id_short')}|{r.get('episode_id')}"
        by_episode.setdefault(ep_key, []).append(float(r["call_model_wall_s"]))

    summary = {
        "unit": "seconds",
        "timed_region": "VLNEvaluator.eval_action call_model wall-clock, with cuda synchronize before/after when CUDA is available",
        "total_steps": len(records),
        "overall": _stats(times),
        "by_step_id": {
            k: _stats(v) for k, v in sorted(by_step.items(), key=lambda kv: int(kv[0]))
        },
        "by_episode": {
            k: {
                **_stats(v),
                "total_s": float(sum(v)),
            }
            for k, v in sorted(by_episode.items())
        },
    }
    _atomic_write_json(os.path.join(output_path, filename), summary)


def _clean_instruction_text(value: Any) -> str:
    """Normalize dataset instructions before logging, prompting, and saving."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return " ".join(
        str(value).replace("\r\n", "\n").replace("\r", "\n").split()
    )


def _legacy_jsonl_to_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    """旧版 JSONL（扁平 success/spl/os/ne）转为 MemoryVLN 使用的嵌套 metrics 格式。"""
    if "metrics" in obj:
        record = dict(obj)
        if "instruction" in record:
            record["instruction"] = _clean_instruction_text(record.get("instruction"))
        return record
    return {
        "scene_id": obj.get("scene_id"),
        "episode_id": obj.get("episode_id"),
        "trajectory_id": obj.get("trajectory_id"),
        "instruction": _clean_instruction_text(
            obj.get("episode_instruction") or obj.get("instruction")
        ),
        "steps": obj.get("steps"),
        "metrics": {
            "success": obj.get("success"),
            "spl": obj.get("spl"),
            "oracle_success": obj.get("os"),
            "distance_to_goal": obj.get("ne"),
            "ndtw": obj.get("ndtw"),
            "sdtw": obj.get("sdtw"),
        },
    }


def _record_has_metrics(rec: Dict[str, Any], required_metrics: set) -> bool:
    if not required_metrics:
        return True
    metrics = rec.get("metrics")
    return isinstance(metrics, dict) and all(
        metrics.get(name) is not None for name in required_metrics
    )


def _load_done_keys_from_result_file(path: str) -> set:
    """从已存在的结果文件恢复已完成 episode 的 key（JSON 数组或旧版 JSONL）。"""
    done: set = set()
    if not os.path.isfile(path):
        return done
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(8192)
        if not head.strip():
            return done
        if head.lstrip().startswith("["):
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                for rec in prev:
                    if isinstance(rec, dict):
                        done.add(_episode_record_key(rec))
            return done
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(res, dict):
                    continue
                if "metrics" in res:
                    done.add(_episode_record_key(res))
                elif "episode_id" in res:
                    done.add(
                        f"{res.get('scene_id')}|{res.get('episode_id')}|{res.get('trajectory_id', '')}"
                    )
    except Exception as e:
        print(f"[warn] 无法读取已有结果 {path}: {e}", flush=True)
    return done


def _load_result_records_from_file(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return records
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(8192)
        if not head.strip():
            return records
        if head.lstrip().startswith("["):
            with open(path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, list):
                for rec in prev:
                    if isinstance(rec, dict):
                        records.append(_legacy_jsonl_to_record(rec))
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(res, dict):
                    continue
                records.append(_legacy_jsonl_to_record(res))
    except Exception as e:
        print(f"[warn] 无法读取已有结果 {path}: {e}", flush=True)
        return []
    return records


def _load_resume_records(result_path: str, legacy_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    candidate_paths = [result_path]
    if legacy_path and legacy_path != result_path:
        candidate_paths.append(legacy_path)
    candidate_paths.extend(_all_partial_result_paths(result_path))
    for path in candidate_paths:
        for rec in _load_result_records_from_file(path):
            by_key[_episode_record_key(rec)] = rec
    return by_key


def _xz_from_agent_state(agent_state) -> tuple:
    pos = agent_state.position
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    pos = np.asarray(pos, dtype=np.float64)
    return float(pos[0]), float(pos[2])


def _yaw_from_agent_state(agent_state) -> float:
    """绕世界 Y 轴朝向（弧度），与四元数一致；用于平面相对位姿。"""
    rot = agent_state.rotation
    if isinstance(rot, quaternion.quaternion):
        q = rot
        try:
            from scipy.spatial.transform import Rotation as R

            r = R.from_quat([q.x, q.y, q.z, q.w])
            euler = r.as_euler("xyz", degrees=False)
            return float(euler[1])
        except ImportError:
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return float(math.atan2(siny_cosp, cosy_cosp))
    euler = np.asarray(rot, dtype=np.float64)
    if euler.size == 4:
        try:
            from scipy.spatial.transform import Rotation as R

            euler = R.from_quat(euler).as_euler("xyz", degrees=False)
        except ImportError:
            euler = euler[:3]
    return float(euler[1])


def _relative_pose_wrt_ref_training_format(ref_state, cur_state) -> np.ndarray:
    """
    以 ref（本 episode 起始观测时刻的 agent state）为原点，将 cur 的位姿变到 ref 的平面局部系：
    与训练 dict 一致 [x,0,z,0,yaw,0]（yaw 为相对航向）。
    水平位移用世界系 xz 差分再乘 R(-yaw_ref)；不使用动作积分。
    """
    x0, z0 = _xz_from_agent_state(ref_state)
    x1, z1 = _xz_from_agent_state(cur_state)
    yaw0 = _yaw_from_agent_state(ref_state)
    yaw1 = _yaw_from_agent_state(cur_state)
    dx = x1 - x0
    dz = z1 - z0
    x_local = dx * math.cos(yaw0) - dz * math.sin(yaw0)
    z_local = dx * math.sin(yaw0) + dz * math.cos(yaw0)
    dyaw = yaw1 - yaw0
    dyaw = math.fmod(dyaw + math.pi, 2 * math.pi) - math.pi
    return np.array([x_local, 0.0, z_local, 0.0, dyaw, 0.0], dtype=np.float32)


def _compute_shrink_k_target(
    n_frames: int,
    adaptive_offset: float = 4.0,
    adaptive_coef: float = 2.0,
    min_frames: int = 16,
) -> int:
    """与 build_vlnce_relative_pose_dataset.compute_shrink_k_target 一致。"""
    adaptive_k = min(n_frames, int(adaptive_offset + adaptive_coef * (n_frames**0.5)))
    return max(adaptive_k, min_frames)


def _adaptive_frame_budget(total_frames: int, args: argparse.Namespace) -> int:
    """历史帧预算 K_target；可选 --max_image_frames>0 作为硬上限。"""
    if total_frames <= 0:
        return 0
    k = _compute_shrink_k_target(
        total_frames,
        adaptive_offset=getattr(args, "shrink_adaptive_offset", 4.0),
        adaptive_coef=getattr(args, "shrink_adaptive_coef", 2.0),
        min_frames=getattr(args, "shrink_min_frames", 16),
    )
    cap = getattr(args, "max_image_frames", -1)
    if cap is not None and cap > 0:
        k = min(k, cap)
    return k


def _select_eval_frame_indices(
    total_frames: int, args: argparse.Namespace, rng=None
) -> List[int]:
    """
    与训练 TSV shrink 一致的全序列两级艾宾浩斯历史帧抽样。
    --skip_frame_sampling：保留全部历史帧。
    """
    if total_frames <= 0:
        return []
    if getattr(args, "skip_frame_sampling", False):
        return list(range(total_frames))

    start_thresh = int(getattr(args, "shrink_start_frames", 16))
    if total_frames <= start_thresh:
        return list(range(total_frames))
    k_target = _adaptive_frame_budget(total_frames, args)
    if total_frames <= k_target:
        return list(range(total_frames))
    return two_level_ebbinghaus_indices(
        num_frames=total_frames,
        max_frames=k_target,
        chunk_size=getattr(args, "ebbinghaus_chunk_size", 8),
        s_inner=getattr(args, "ebbinghaus_s", 5.0),
        s_outer=getattr(args, "ebbinghaus_s_outer", 3.0),
        rng=rng,
    )


def _vit_frame_budget_indices(
    sel_full: List[int], vit_cap: int, strategy: str
) -> List[int]:
    """
    在已有时间有序索引序列 ``sel_full``（通常为艾宾浩斯选帧结果）上再做一层额度裁剪。
    用于可选地进一步压低 ViT 输入帧数；VGGT 若走「全帧」路径则不受此函数影响。
    """
    if not sel_full:
        return []
    if vit_cap <= 0 or len(sel_full) <= vit_cap:
        return list(sel_full)
    st = (strategy or "tail").lower()
    if st == "uniform":
        n = len(sel_full)
        if vit_cap == 1:
            return [sel_full[n // 2]]
        raw_idx = np.linspace(0, n - 1, vit_cap)
        pick_j = sorted({int(round(float(x))) for x in raw_idx})
        return [sel_full[j] for j in pick_j]
    return sel_full[-vit_cap:]


def _parse_nav_action(raw: str, actions2idx: dict) -> int:
    if not raw:
        return actions2idx["STOP"][0]
    upper = raw.upper()
    for name in ("STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
        if name in upper:
            return actions2idx[name][0]
    return actions2idx["STOP"][0]


def register_eval_aligned_frame_sampling_args(parser: argparse.ArgumentParser) -> None:
    """VLNEvaluator.eval_action 使用的历史帧采样 / call_model 侧 CLI。"""
    parser.add_argument("--max_steps", default=400, type=int, help="max_steps")
    parser.add_argument(
        "--max_image_frames",
        type=int,
        default=-1,
        help=">0 时对 K_target 再取 min(K, 此值)；-1 表示仅用自适应公式。",
    )
    parser.add_argument(
        "--skip_frame_sampling",
        action="store_true",
        help="关闭历史抽帧，使用全部历史帧（EVAL_SKIP_FRAME_SAMPLING=1）。",
    )
    parser.add_argument(
        "--shrink_start_frames",
        type=int,
        default=16,
        help="历史帧 N<=该值时不抽帧（与训练 --shrink_start_frames 一致）。",
    )
    parser.add_argument(
        "--shrink_adaptive_offset",
        type=float,
        default=4.0,
        help="K_target 常数项 offset + coef*sqrt(N)。",
    )
    parser.add_argument(
        "--shrink_adaptive_coef",
        type=float,
        default=2.0,
        help="K_target sqrt 系数。",
    )
    parser.add_argument(
        "--shrink_min_frames",
        type=int,
        default=16,
        help="K_target 下限 max(..., min_frames)。",
    )
    parser.add_argument("--ebbinghaus_chunk_size", type=int, default=8)
    parser.add_argument(
        "--ebbinghaus_s",
        type=float,
        default=5.0,
        help="艾宾浩斯内层 s（与 build_vlnce_relative_pose_dataset 默认 5 对齐）。",
    )
    parser.add_argument(
        "--ebbinghaus_s_outer",
        type=float,
        default=3.0,
        help="艾宾浩斯外层 s（与 build_vlnce_relative_pose_dataset 默认 3 对齐）。",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--split_vit_vggt_frames",
        action="store_true",
        help=(
            "ViT/chat/pixel_values 使用艾宾浩斯双层抽帧（与默认评测 shrink 一致）；"
            "VGGT 使用当前步为止的全部历史仿真帧（不抽帧）。可选 --vit_frame_cap>0 在艾宾浩斯结果上再截 ViT。"
            "依赖模型 forward 的 agent_poses_vggt。"
        ),
    )
    parser.add_argument(
        "--vit_frame_cap",
        type=int,
        default=0,
        help=(
            "与 --split_vit_vggt_frames 连用：在艾宾浩斯选帧结果上再限制 ViT 最多帧数（0=不额外限制，仅艾宾浩斯）。"
        ),
    )
    parser.add_argument(
        "--vit_sel_strategy",
        type=str,
        default="tail",
        choices=("tail", "uniform"),
        help="仅 vit_frame_cap>0 时：在艾宾浩斯索引序列上 tail（末尾）或 uniform（均匀）再取样。",
    )


def register_eval_phased_memory_args(parser: argparse.ArgumentParser) -> None:
    """分阶段评测：<memory_start_frames 前 VGGT 8+48 跨步 KV；之后帧级 Memory、每步清空 VGGT KV。"""
    parser.add_argument(
        "--eval_phased_memory",
        action="store_true",
        help=(
            "开启分阶段评测（EVAL_PHASED_MEMORY=1）：历史帧数 < --memory_start_frames 时 "
            "不启用帧级 Memory、VGGT 保留跨步 KV 并 8+48 trim；达到阈值后启用 Memory、每步清空 VGGT KV。"
            "默认关闭，行为与改前一致。"
        ),
    )
    parser.add_argument(
        "--memory_start_frames",
        type=int,
        default=32,
        help="与 --eval_phased_memory 连用：累积仿真 RGB 帧数达到该值后启用帧级 Memory（默认 32）。",
    )


def eval_phased_step_flags(args: argparse.Namespace, total_frames: int) -> Dict[str, Any]:
    """
  返回本步 phased 开关。未开启 --eval_phased_memory 时与历史评测一致：
  每步清空 VGGT KV；帧级 Memory 由 chunk_meta.enable_frame_memory=True 控制。
    """
    if not getattr(args, "eval_phased_memory", False):
        return {
            "phased_active": False,
            "enable_frame_memory": True,
            "reset_vggt_kv": True,
            "force_vggt_trim": None,
        }
    start = int(getattr(args, "memory_start_frames", 32))
    before_memory = total_frames < start
    return {
        "phased_active": True,
        "enable_frame_memory": not before_memory,
        "reset_vggt_kv": not before_memory,
        "force_vggt_trim": before_memory,
        "memory_start_frames": start,
    }


def register_memoryvln_rollout_model_args(parser: argparse.ArgumentParser) -> None:
    """评测使用的 MemoryVLNInference CLI（位姿编码、VGGT KV、Memory/Scene/置信度）。"""
    parser.add_argument(
        "--no_position_encoding",
        action="store_true",
        help="Disable agent pose injection (VGGT + ViT sine PE). Default: enabled（首帧为参考的仿真器相对位姿 6D）.",
    )
    parser.add_argument(
        "--vggt_kv_trim_8_48",
        action="store_true",
        help=(
            "评测时对 VGGT past_key_values 启用经典 8+48（头+尾）截断；默认关闭。"
            "与 --no_vggt_kv_trim 同时出现时关闭截断。"
        ),
    )
    parser.add_argument(
        "--no_vggt_kv_trim",
        action="store_true",
        help="强制关闭 VGGT KV 截断（覆盖 --vggt_kv_trim_8_48）。",
    )
    _mem = parser.add_mutually_exclusive_group()
    _mem.add_argument(
        "--use_memory_bank",
        action="store_true",
        help="覆盖 checkpoint：强制开启显式记忆库（需权重与结构一致，否则可能缺失 lam_mem 等）。",
    )
    _mem.add_argument(
        "--no_use_memory_bank",
        action="store_true",
        help="覆盖 checkpoint：强制关闭显式记忆库（忽略 config.json 中的 use_memory_bank）。",
    )
    parser.add_argument(
        "--max_memory_frames",
        type=int,
        default=16,
        help="帧级 Memory 总容量，默认与 4 个短期槽位 + 12 个长期槽位一致。",
    )
    parser.add_argument(
        "--memory_short_capacity",
        type=int,
        default=4,
        help="帧级 Memory 短期槽位数量。",
    )
    parser.add_argument(
        "--memory_long_capacity",
        type=int,
        default=12,
        help="帧级 Memory 长期槽位数量。",
    )
    _mcomp = parser.add_mutually_exclusive_group()
    _mcomp.add_argument(
        "--use_memory_compression",
        action="store_true",
        help="覆盖 checkpoint：记忆模块后对多帧做 compress_to_memory_tokens 并裁剪 image_pad。",
    )
    _mcomp.add_argument(
        "--no_memory_compression",
        action="store_true",
        help="覆盖 checkpoint：只做记忆融合 + lam_mem，不做帧压缩与序列裁剪。",
    )
    _scene = parser.add_mutually_exclusive_group()
    _scene.add_argument(
        "--use_scene_memory_bank",
        action="store_true",
        help="覆盖 checkpoint：开启跨场景 Scene Bank（检索/warm/写回；需 use_memory_bank）。",
    )
    _scene.add_argument(
        "--no_use_scene_memory_bank",
        action="store_true",
        help="覆盖 checkpoint：关闭跨场景 Scene Bank。",
    )
    _fconf = parser.add_mutually_exclusive_group()
    _fconf.add_argument(
        "--use_frame_confidence",
        action="store_true",
        help="覆盖 checkpoint：开启帧级置信度门控（融合 eff_ratio 与 Scene 写回门槛）。",
    )
    _fconf.add_argument(
        "--no_use_frame_confidence",
        action="store_true",
        help="覆盖 checkpoint：关闭帧级置信度门控。",
    )


def build_memoryvln_inference_from_args(args: argparse.Namespace) -> "MemoryVLNInference":
    """与 evaluation.eval() 一致的 Memory/Scene/置信度等覆盖后构造推理模型。"""
    local_rank = args.local_rank

    _mb_override = None
    if getattr(args, "use_memory_bank", False):
        _mb_override = True
    elif getattr(args, "no_use_memory_bank", False):
        _mb_override = False

    _mc_override = None
    if getattr(args, "use_memory_compression", False):
        _mc_override = True
    elif getattr(args, "no_memory_compression", False):
        _mc_override = False

    _scene_override = None
    if getattr(args, "use_scene_memory_bank", False):
        _scene_override = True
    elif getattr(args, "no_use_scene_memory_bank", False):
        _scene_override = False

    _fc_override = None
    if getattr(args, "use_frame_confidence", False):
        _fc_override = True
    elif getattr(args, "no_use_frame_confidence", False):
        _fc_override = False

    max_memory_frames = int(getattr(args, "max_memory_frames", 16))
    memory_short_capacity = int(getattr(args, "memory_short_capacity", 4))
    memory_long_capacity = int(getattr(args, "memory_long_capacity", 12))
    if memory_short_capacity <= 0 or memory_long_capacity <= 0:
        raise ValueError("memory_short_capacity and memory_long_capacity must be positive")
    if max_memory_frames != memory_short_capacity + memory_long_capacity:
        raise ValueError(
            "max_memory_frames must equal memory_short_capacity + memory_long_capacity "
            f"({max_memory_frames} != {memory_short_capacity} + {memory_long_capacity})"
        )

    return MemoryVLNInference(
        args.model_path,
        processor_path=args.processor_path,
        device=f"cuda:{local_rank}",
        use_position_encoding=not getattr(args, "no_position_encoding", False),
        vggt_trim_kv_cache_evaluation=(
            getattr(args, "vggt_kv_trim_8_48", False)
            and not getattr(args, "no_vggt_kv_trim", False)
        ),
        memory_bank_override=_mb_override,
        memory_compression_override=_mc_override,
        scene_memory_bank_override=_scene_override,
        frame_confidence_override=_fc_override,
        max_memory_frames=max_memory_frames,
        memory_short_capacity=memory_short_capacity,
        memory_long_capacity=memory_long_capacity,
    )


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device(f"cuda:{args.local_rank}")
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(self.output_path, exist_ok=True)
        self.epoch = epoch
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.save_video_ratio = args.save_video_ratio


        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            if args is not None and getattr(args, "habitat_data_path", None):
                self.config.habitat.dataset.data_path = args.habitat_data_path
            measurements = self.config.habitat.task.measurements
            if "ndtw" in measurements:
                dataset_path = str(self.config.habitat.dataset.data_path).replace(
                    "{split}", self.split
                )
                gt_path = os.path.join(
                    os.path.dirname(dataset_path),
                    f"{self.split}_guide_gt.json.gz",
                )
                measurements["ndtw"] = measures.NDTWMeasurementConfig(
                    gt_path=gt_path,
                    success_distance=3.0,
                )
            if "sdtw" in measurements:
                measurements["sdtw"] = measures.SDTWMeasurementConfig()
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.image_processor = model.processor
        self.model = model
        self.tokenizer = model.tokenizer
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "MOVE_FORWARD": [1],
            "TURN_LEFT": [2],
            "TURN_RIGHT": [3]
        })

        self.num_history = args.num_history


    def config_env(self) -> Env:
        env = Env(config=self.config)
        return env

    def _episode_skip_keys(self, episode) -> List[str]:
        tid = getattr(episode, "trajectory_id", None)
        scene_id_short = os.path.splitext(
            os.path.basename(str(episode.scene_id).replace("\\", "/"))
        )[0]
        return [
            _episode_obj_key(episode),
            f"{scene_id_short}|{episode.episode_id}|{tid}",
            f"{scene_id_short}|{episode.episode_id}|",
        ]

    def eval_action(self, idx) -> None:
        args = self.args
        result_path = os.path.join(self.output_path, args.result_json)
        live_result = not getattr(args, "no_live_result", False)
        partial_path = _partial_result_path(result_path, idx) if live_result else ""

        legacy_path = os.path.join(self.output_path, "result.json")
        resume_records = _load_resume_records(result_path, legacy_path)
        configured_measurements = self.config.habitat.task.measurements
        required_resume_metrics = {
            name for name in ("ndtw", "sdtw") if name in configured_measurements
        }
        if required_resume_metrics:
            before = len(resume_records)
            resume_records = {
                k: rec
                for k, rec in resume_records.items()
                if _record_has_metrics(rec, required_resume_metrics)
            }
            missing = before - len(resume_records)
            if missing and get_rank() == 0:
                print(
                    f"[eval] {missing} 条已有结果缺少 "
                    f"{sorted(required_resume_metrics)}，将重新评测以补齐指标",
                    flush=True,
                )
        done_keys = set(resume_records.keys())

        local_results: List[Dict[str, Any]] = []
        local_profile_records: List[Dict[str, Any]] = []
        profile_speed = bool(getattr(args, "profile_inference_speed", False))
        profile_jsonl_path = (
            os.path.join(self.output_path, f"inference_profile_rank{idx}.jsonl")
            if profile_speed
            else ""
        )
        if profile_jsonl_path:
            os.makedirs(self.output_path, exist_ok=True)
            open(profile_jsonl_path, "w", encoding="utf-8").close()

        if live_result and partial_path and os.path.isfile(partial_path):
            try:
                with open(partial_path, "r", encoding="utf-8") as f:
                    part = json.load(f)
                if isinstance(part, list):
                    for r in part:
                        if not _record_has_metrics(r, required_resume_metrics):
                            continue
                        k = _episode_record_key(r)
                        if k in resume_records:
                            local_results.append(copy.deepcopy(resume_records[k]))
                        elif k not in done_keys:
                            local_results.append(r)
                            done_keys.add(k)
            except Exception as e:
                print(f"[warn] 无法读取增量结果 {partial_path}: {e}", flush=True)

        env = self.config_env()
        # 续跑时先汇总所有已完成样本，只对剩余 episode 重新按当前 world size 均分。
        all_eps = list(env.episodes)
        remaining_eps = [
            ep for ep in all_eps
            if not any(k in done_keys for k in self._episode_skip_keys(ep))
        ]
        eval_limit = int(getattr(args, "eval_limit_episodes", 0) or 0)
        if eval_limit > 0:
            remaining_eps = remaining_eps[:eval_limit]
        my_eps = [remaining_eps[j] for j in range(len(remaining_eps)) if j % self.env_num == idx]
        env.episodes = my_eps

        pbar = tqdm.tqdm(my_eps, desc=f"rank{idx}-eval", dynamic_ncols=True)
        for episode in pbar:
            scene_id_short = os.path.splitext(
                os.path.basename(str(episode.scene_id).replace("\\", "/"))
            )[0]
            if "objectnav" in self.config_path:
                episode_instruction = episode.object_category
            else:
                inst = episode.instruction.instruction_text
                if isinstance(inst, list):
                    episode_instruction = inst[0] if inst else ""
                else:
                    episode_instruction = inst
            episode_instruction = _clean_instruction_text(episode_instruction)
            print("episode start: ", episode_instruction)
            episode_id = episode.episode_id

            env.current_episode = episode
            observations = env.reset()

            vis_frames = []
            step_id = 0

            should_save_video = self.save_video and (random.random() < self.save_video_ratio)
            if should_save_video:
                os.makedirs(os.path.join(self.output_path, f"vis_{self.epoch}"), exist_ok=True)

            rgb_list = []
            agent_states_list = []
            self.model.model.past_key_values_vggt = None
            _phased_logged = False

            while not env.episode_over:
                rgb = observations["rgb"]
                image = Image.fromarray(rgb).convert("RGB")
                rgb_list.append(image)
                agent_states_list.append(env.sim.get_agent_state())

                info = env.get_metrics()

                total_frames = len(rgb_list)
                rng = np.random.default_rng(int(self.args.seed) + step_id + idx * 1_000_003)
                sel_ebb = _select_eval_frame_indices(
                    total_frames, self.args, rng=rng
                )
                split_vv = getattr(self.args, "split_vit_vggt_frames", False)
                if split_vv:
                    # ViT：艾宾浩斯双层抽帧（与训练 shrink 一致）；可选再在 sel_ebb 上二次裁剪 vit_frame_cap
                    sel_vit = sel_ebb
                    vit_cap = int(getattr(self.args, "vit_frame_cap", 0))
                    if vit_cap > 0 and len(sel_vit) > vit_cap:
                        vit_strat = getattr(self.args, "vit_sel_strategy", "tail")
                        sel_vit = _vit_frame_budget_indices(
                            sel_vit, vit_cap, vit_strat
                        )
                    # VGGT：本步为止的全部仿真帧（不抽帧）
                    sel_vggt = list(range(total_frames))
                else:
                    sel_vit = sel_ebb
                    sel_vggt = None

                images_vit = [rgb_list[i] for i in sel_vit]
                images_vggt = (
                    [rgb_list[i] for i in sel_vggt] if split_vv else None
                )

                inner = self.model.model
                use_pe = getattr(inner.config, "use_position_encoding", True)
                device = inner.device
                agent_poses_vggt = None
                if use_pe:
                    dtype = next(inner.visual.parameters()).dtype
                    ref_state = agent_states_list[0]
                    origin6 = np.zeros(6, dtype=np.float32)

                    def _poses_for(indices: List[int]) -> torch.Tensor:
                        rows = []
                        for i in indices:
                            if i == 0:
                                rows.append(origin6)
                            else:
                                rows.append(
                                    _relative_pose_wrt_ref_training_format(
                                        ref_state, agent_states_list[i]
                                    )
                                )
                        return torch.tensor(
                            np.stack(rows, axis=0), device=device, dtype=dtype
                        )

                    agent_poses = _poses_for(sel_vit)
                    if split_vv:
                        agent_poses_vggt = _poses_for(sel_vggt)
                else:
                    agent_poses = None
                phased_flags = eval_phased_step_flags(self.args, total_frames)
                if phased_flags["phased_active"] and not _phased_logged and get_rank() == 0:
                    print(
                        f"[eval][phased] memory_start_frames={phased_flags['memory_start_frames']} "
                        f"(<N: VGGT KV retain+trim; >=N: frame Memory, VGGT KV reset/step)",
                        flush=True,
                    )
                    _phased_logged = True
                if (
                    phased_flags["phased_active"]
                    and get_rank() == 0
                    and total_frames == phased_flags["memory_start_frames"]
                ):
                    print(
                        f"[eval][phased] episode {episode_id} step {step_id}: "
                        f"reach {total_frames} frames -> enable frame Memory, reset VGGT KV",
                        flush=True,
                    )
                # split_vit_vggt：ViT=艾宾浩斯（可选 vit_frame_cap 二次裁剪）；VGGT=全程全部帧 + agent_poses_vggt
                if profile_speed and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _profile_t0 = time.perf_counter()
                raw_action = self.model.call_model(
                    images_vit,
                    episode_instruction,
                    step_id,
                    episode_id=str(episode_id),
                    agent_poses=agent_poses,
                    observations_vggt=images_vggt,
                    agent_poses_vggt=agent_poses_vggt,
                    reset_vggt_kv=phased_flags["reset_vggt_kv"],
                    force_vggt_trim=phased_flags["force_vggt_trim"],
                    enable_frame_memory=phased_flags["enable_frame_memory"],
                )[0]
                if profile_speed and torch.cuda.is_available():
                    torch.cuda.synchronize()
                _profile_elapsed = time.perf_counter() - _profile_t0
                action = _parse_nav_action(raw_action, self.actions2idx)

                if profile_speed:
                    profile_rec = {
                        "rank": int(idx),
                        "scene_id_short": scene_id_short,
                        "episode_id": str(episode_id),
                        "trajectory_id": getattr(episode, "trajectory_id", None),
                        "step_id": int(step_id),
                        "total_frames": int(total_frames),
                        "vit_frames": int(len(images_vit)),
                        "vggt_frames": int(len(images_vggt) if images_vggt is not None else 0),
                        "call_model_wall_s": float(_profile_elapsed),
                        "steps_per_s": float(1.0 / _profile_elapsed) if _profile_elapsed > 0 else 0.0,
                        "raw_action": str(raw_action),
                        "parsed_action": int(action),
                    }
                    local_profile_records.append(profile_rec)
                    with open(profile_jsonl_path, "a", encoding="utf-8") as pf:
                        pf.write(json.dumps(profile_rec, ensure_ascii=False) + "\n")
                    pbar.set_postfix(call_s=f"{_profile_elapsed:.3f}")

                if info["top_down_map"] is not None and should_save_video:
                    frame = observations_to_image({"rgb": observations["rgb"]}, info)
                    vis_frames.append(frame)

                if step_id >= self.args.max_steps:
                    action = 0

                observations = env.step(action)
                step_id += 1

            metrics = env.get_metrics()
            if should_save_video:
                images_to_video(
                    vis_frames,
                    os.path.join(self.output_path, f"vis_{self.epoch}"),
                    f"{scene_id_short}_{episode_id}",
                    fps=6,
                    quality=9,
                )
            vis_frames.clear()
            trajectory_metrics = ""
            if "ndtw" in metrics:
                trajectory_metrics += f", ndtw: {metrics['ndtw']}"
            if "sdtw" in metrics:
                trajectory_metrics += f", sdtw: {metrics['sdtw']}"
            print(
                f"scene_episode {scene_id_short}_{episode_id} success: {metrics['success']}, "
                f"spl: {metrics['spl']}, os: {metrics['oracle_success']}, "
                f"ne: {metrics['distance_to_goal']}{trajectory_metrics}"
            )

            rec = {
                "scene_id": episode.scene_id,
                "episode_id": episode_id,
                "trajectory_id": getattr(episode, "trajectory_id", None),
                "instruction": episode_instruction,
                "steps": step_id,
                "metrics": _metrics_to_jsonable(metrics),
            }
            local_results.append(rec)
            done_keys.add(_episode_record_key(rec))
            if live_result and partial_path:
                _atomic_write_json(partial_path, local_results)

        env.close()

        ws = get_world_size()
        if ws > 1:
            gathered: List[Optional[List[Dict[str, Any]]]] = [None] * ws
            dist.all_gather_object(gathered, local_results)
            profile_gathered: List[Optional[List[Dict[str, Any]]]] = [None] * ws
            if profile_speed:
                dist.all_gather_object(profile_gathered, local_profile_records)
            if get_rank() != 0:
                return
            flat: List[Dict[str, Any]] = []
            for g in gathered:
                if g:
                    flat.extend(g)
            local_results = flat
            if profile_speed:
                profile_flat: List[Dict[str, Any]] = []
                for g in profile_gathered:
                    if g:
                        profile_flat.extend(g)
                local_profile_records = profile_flat

        by_key: Dict[str, Dict[str, Any]] = dict(resume_records)
        for r in local_results:
            by_key[_episode_record_key(r)] = r

        final_list = sorted(
            by_key.values(),
            key=lambda x: (str(x.get("scene_id", "")), str(x.get("episode_id", ""))),
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_list, f, indent=2, ensure_ascii=False)

        if live_result:
            _remove_partial_result_files(result_path)

        if final_list:
            n = len(final_list)
            succ = sum(1 for r in final_list if r.get("metrics", {}).get("success"))
            spls = [r["metrics"].get("spl", 0.0) for r in final_list if "metrics" in r]
            oss = [r["metrics"].get("oracle_success", 0.0) for r in final_list if "metrics" in r]
            nes = [r["metrics"].get("oracle_navigation_error", 0.0) for r in final_list if "metrics" in r]
            mean_spl = float(np.mean(spls)) if spls else 0.0
            mean_os = float(np.mean(oss)) if oss else 0.0
            mean_ne = float(np.mean(nes)) if nes else 0.0
            ndtws = [
                r["metrics"].get("ndtw", 0.0)
                for r in final_list
                if "metrics" in r and r["metrics"].get("ndtw") is not None
            ]
            sdtws = [
                r["metrics"].get("sdtw", 0.0)
                for r in final_list
                if "metrics" in r and r["metrics"].get("sdtw") is not None
            ]
            mean_ndtw = float(np.mean(ndtws)) if ndtws else None
            mean_sdtw = float(np.mean(sdtws)) if sdtws else None
            sr = succ / n
            dtw_summary = ""
            if mean_ndtw is not None:
                dtw_summary += f" nDTW={mean_ndtw:.4f}"
            if mean_sdtw is not None:
                dtw_summary += f" SDTW={mean_sdtw:.4f}"
            print(
                f"[eval] 写入 {result_path} | episodes={n} "
                f"SR={sr:.4f} SPL={mean_spl:.4f} OS={mean_os:.4f} "
                f"NE={mean_ne:.4f}{dtw_summary}",
                flush=True,
            )
            if os.environ.get("REPORT_TO", "").lower() == "wandb" and get_rank() == 0:
                try:
                    import wandb

                    wandb_metrics = {
                        "eval/success_rate": sr,
                        "eval/spl": mean_spl,
                        "eval/oracle_success": mean_os,
                        "eval/oracle_navigation_error": mean_ne,
                        "eval/episodes": n,
                    }
                    if mean_ndtw is not None:
                        wandb_metrics["eval/ndtw"] = mean_ndtw
                    if mean_sdtw is not None:
                        wandb_metrics["eval/sdtw"] = mean_sdtw
                    wandb.log(wandb_metrics)
                except Exception as e:
                    print(f"[warn] wandb.log failed: {e}", flush=True)

        if profile_speed and get_rank() == 0 and local_profile_records:
            _write_inference_profile_summary(self.output_path, local_profile_records)
            _profile_times = [float(r["call_model_wall_s"]) for r in local_profile_records]
            print(
                f"[profile] inference steps={len(_profile_times)} "
                f"mean={np.mean(_profile_times):.4f}s/step "
                f"median={np.median(_profile_times):.4f}s/step "
                f"p90={np.percentile(_profile_times, 90):.4f}s/step "
                f"summary={os.path.join(self.output_path, 'inference_profile_summary.json')}",
                flush=True,
            )




class MemoryVLNInference:
    def __init__(
        self,
        pretrained,
        processor_path=None,
        device="cuda",
        use_position_encoding=True,
        vggt_trim_kv_cache_evaluation: bool = False,
        memory_bank_override: Optional[bool] = None,
        memory_compression_override: Optional[bool] = None,
        scene_memory_bank_override: Optional[bool] = None,
        frame_confidence_override: Optional[bool] = None,
        max_memory_frames: int = 16,
        memory_short_capacity: int = 4,
        memory_long_capacity: int = 12,
    ):
        try:
            from qwen_vl.train.lora_utils import resolve_lora_base_and_adapter

            model_load_path, lora_dir = resolve_lora_base_and_adapter(pretrained)
        except Exception as _lora_resolve_exc:
            model_load_path, lora_dir = pretrained, None
            if get_rank() == 0:
                print(f"[inference] LoRA path resolve skipped: {_lora_resolve_exc}", flush=True)

        proc_src = processor_path or model_load_path
        config_src = pretrained if os.path.isfile(os.path.join(pretrained, "config.json")) else model_load_path
        config = AutoConfig.from_pretrained(config_src)
        if get_rank() == 0 and config_src != model_load_path:
            print(f"[inference] config loaded from LoRA checkpoint: {config_src}", flush=True)
        setattr(config, "use_position_encoding", use_position_encoding)
        setattr(config, "max_memory_frames", max_memory_frames)
        setattr(config, "memory_short_capacity", memory_short_capacity)
        setattr(config, "memory_long_capacity", memory_long_capacity)
        setattr(config, "vggt_trim_kv_cache_evaluation", vggt_trim_kv_cache_evaluation)
        if memory_bank_override is not None:
            setattr(config, "use_memory_bank", memory_bank_override)
            print(
                f"[inference] use_memory_bank CLI override: {memory_bank_override}",
                flush=True,
            )
        if memory_compression_override is not None:
            setattr(config, "use_memory_compression", memory_compression_override)
            print(
                f"[inference] use_memory_compression CLI override: {memory_compression_override}",
                flush=True,
            )
        if scene_memory_bank_override is not None:
            setattr(config, "use_scene_memory_bank", scene_memory_bank_override)
            print(
                f"[inference] use_scene_memory_bank CLI override: {scene_memory_bank_override}",
                flush=True,
            )
        if frame_confidence_override is not None:
            setattr(config, "use_frame_confidence", frame_confidence_override)
            print(
                f"[inference] use_frame_confidence CLI override: {frame_confidence_override}",
                flush=True,
            )
        if get_rank() == 0 and model_load_path != pretrained:
            print(
                f"[inference] LoRA checkpoint detected: load base={model_load_path}; "
                f"adapter={lora_dir}",
                flush=True,
            )

        self.model = Qwen2_5_VLForConditionalGenerationForMemoryVLN.from_pretrained(
            model_load_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode='evaluation'
        ).eval()

        try:
            if lora_dir is not None:
                from peft import PeftModel

                self.model.model = PeftModel.from_pretrained(
                    self.model.model,
                    lora_dir,
                    is_trainable=False,
                )
                if get_rank() == 0:
                    print(f"[inference] LoRA adapter loaded from {lora_dir}", flush=True)
        except Exception as _lora_exc:
            if get_rank() == 0:
                print(f"[inference] LoRA load skipped: {_lora_exc}", flush=True)

        if getattr(self.model.config, "use_memory_bank", False) and getattr(
            self.model, "fused_memory", None
        ) is not None:
            _pool = int(os.environ.get("EVAL_MEMORY_POOL_FRAMES", "0"))
            if _pool > 0:
                self.model.fused_memory.configure_inference_memory_pool(_pool)
            _ic = os.environ.get("EVAL_INFERENCE_COMPRESS_MAX_FRAMES", "").strip()
            if _ic:
                self.model.fused_memory.inference_compress_max_frames = int(_ic)
            print(
                "[inference] use_memory_bank=True, chunk_meta 将按步传入；"
                f"EVAL_MEMORY_POOL_FRAMES={_pool or 'off'}, "
                f"EVAL_INFERENCE_COMPRESS_MAX_FRAMES={_ic or 'off'}",
                flush=True,
            )
            if get_rank() == 0:
                fm = self.model.fused_memory
                _cfg_s = int(getattr(self.model.config, "memory_short_capacity", -1))
                _cfg_l = int(getattr(self.model.config, "memory_long_capacity", -1))
                print(
                    "[inference] 帧记忆池容量（实际 fused_memory）: "
                    f"short={getattr(fm, 'short_capacity', '?')}, "
                    f"long={getattr(fm, 'long_capacity', '?')}, "
                    f"max_frames={getattr(fm, 'max_frames', '?')} | "
                    f"config.json: memory_short_capacity={_cfg_s}, "
                    f"memory_long_capacity={_cfg_l}",
                    flush=True,
                )
            if getattr(self.model.config, "use_scene_memory_bank", False):
                print(
                    "[inference] use_scene_memory_bank=True（跨场景检索/写回；需 use_memory_bank）",
                    flush=True,
                )
            if getattr(self.model.config, "use_frame_confidence", False):
                print(
                    "[inference] use_frame_confidence=True（帧级置信度门控融合；Scene 写回可拒低置信）",
                    flush=True,
                )

        self.tokenizer = AutoTokenizer.from_pretrained(proc_src, padding_side="left")
        self.processor = AutoProcessor.from_pretrained(proc_src, padding_side="left")
        
        self.device = device


    def call_model(
        self,
        observations,
        task,
        step_id,
        episode_id: Optional[str] = None,
        add_frame_index: bool = False,
        gen_kwargs: dict = {},
        agent_poses=None,
        observations_vggt=None,
        agent_poses_vggt=None,
        reset_vggt_kv: bool = True,
        force_vggt_trim: Optional[bool] = None,
        enable_frame_memory: bool = True,
    ):
        
        messages = []
        message = [
                {"role": "system", 
                "content": "You are a visual language navigation model, and your should go to the locations to complete the given task. Compare the observation and instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location and finish the task."
                }
            ]
        context = f"These images are your historical observations and your current observation.\n Your task is to {task} \n You should take one of the following actions:\n MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        for i in enumerate([task]):
    
            visual = observations
            if isinstance(visual, Image.Image): 
                message.append({"role": "user", "content": [{"type": "image", "image": visual}, {"type": "text", "text": context}]})
            elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  
                image_content = []
                image_count = 0
                for v in visual:
                    if add_frame_index:
                        image_content.append({"type": "text", "text": "Frame-{}: ".format(image_count)})    
                    image_content.append({"type": "image", "image": v})
                    image_count += 1
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
        images_vggt = []
        image_inputs = []
        for message in messages:
            vision_info = extract_vision_info(message)
            cur_images_vggt = []
            for _, ele in enumerate(vision_info):
                if "image" not in ele:
                    continue
                image = ele["image"]
                if isinstance(image, Image.Image):
                    pass
                elif isinstance(image, str) and "base64," in image:
                    _, base64_data = image.split("base64,", 1)
                    data = base64.b64decode(base64_data)
                    with BytesIO(data) as bio:
                        image = copy.deepcopy(Image.open(bio))
                else:
                    raise NotImplementedError("Unsupported image type")

                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image = load_and_preprocess_images([image])[0]

                _, height, width = image.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image = image[:, :height, :width]
                image_inputs.append(image)
                cur_images_vggt.append(image)

            images_vggt.append(torch.stack(cur_images_vggt))
        
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False
        )
        device = self.model.device

        if observations_vggt is not None:
            cur_images_vggt = []
            for image in observations_vggt:
                assert isinstance(image, Image.Image), f"Unsupported image type: {type(image)}"
                image_t = load_and_preprocess_images([image])[0]
                _, height, width = image_t.shape
                if (width // patch_size) % merge_size > 0:
                    width = width - (width // patch_size) % merge_size * patch_size
                if (height // patch_size) % merge_size > 0:
                    height = height - (height // patch_size) % merge_size * patch_size
                image_t = image_t[:, :height, :width]
                cur_images_vggt.append(image_t)
            images_vggt = [torch.stack(cur_images_vggt)]

        inputs["images_vggt"] = [feat.to(device) for feat in images_vggt]
        inputs = inputs.to(device)
    
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        
        pad_token_id = self.tokenizer.pad_token_id
        generate_kwargs = {
            **inputs,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "do_sample": True if gen_kwargs["temperature"] > 0 else False,
            "temperature": gen_kwargs["temperature"],
            "top_p": gen_kwargs["top_p"],
            "num_beams": gen_kwargs["num_beams"],
            "max_new_tokens": gen_kwargs["max_new_tokens"],
        }
        if agent_poses is not None:
            generate_kwargs["agent_poses"] = agent_poses
        if agent_poses_vggt is not None:
            generate_kwargs["agent_poses_vggt"] = agent_poses_vggt

        use_mb = bool(getattr(self.model.config, "use_memory_bank", False))
        use_scene = bool(getattr(self.model.config, "use_scene_memory_bank", False))
        if use_mb or use_scene:
            generate_kwargs["chunk_meta"] = {
                "is_first_chunk": step_id == 0,
                "is_last_step": False,
                "episode_id": episode_id,
                "enable_frame_memory": bool(enable_frame_memory),
            }

        _prev_vggt_trim = bool(
            getattr(self.model.config, "vggt_trim_kv_cache_evaluation", False)
        )
        if force_vggt_trim is not None:
            setattr(
                self.model.config,
                "vggt_trim_kv_cache_evaluation",
                bool(force_vggt_trim),
            )
        try:
            if reset_vggt_kv:
                # 默认：每步清空 VGGT KV（与训练每样本内多帧一致）。phased 早期阶段保留跨步 KV。
                self.model.past_key_values_vggt = None
            cont = self.model.generate(**generate_kwargs)
        finally:
            if force_vggt_trim is not None:
                setattr(
                    self.model.config,
                    "vggt_trim_kv_cache_evaluation",
                    _prev_vggt_trim,
                )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return answers




   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--processor_path", type=str, default=None)
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument(
        "--habitat_data_path",
        type=str,
        default=None,
        help="覆盖 Habitat dataset.data_path（可含占位 {split}，与官方 json 模板一致）；"
        "例：RxR 仅英语评测 data/datasets/rxr/{split}/{split}_guide_en.json.gz",
    )
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/memoryvln')
    parser.add_argument(
        "--result_json",
        type=str,
        default="result.json",
        help="相对于 output_path 的结果文件名，如 result_val_unseen.json；已存在则跳过已完成 episode 并合并写入",
    )
    parser.add_argument(
        "--no_live_result",
        action="store_true",
        help="关闭实时落盘（默认每完成一条 episode 写入 *.partial_rank{k}.json）",
    )
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument("--save_video_ratio", type=float, default=0.05, help="0~1")
    parser.add_argument(
        "--eval_limit_episodes",
        type=int,
        default=0,
        help="只评测当前 split 前 N 个未完成 episode；0 表示不限制。",
    )
    parser.add_argument(
        "--profile_inference_speed",
        action="store_true",
        help="记录每个导航步 call_model 的 wall-clock 耗时，并写出 inference_profile_*.jsonl/summary。",
    )
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    register_eval_aligned_frame_sampling_args(parser)
    register_eval_phased_memory_args(parser)
    register_memoryvln_rollout_model_args(parser)

    
    args = parser.parse_args()
    if getattr(args, "eval_phased_memory", False) and get_rank() == 0:
        print(
            f"[evaluation] eval_phased_memory=on memory_start_frames="
            f"{getattr(args, 'memory_start_frames', 32)}",
            flush=True,
        )
    set_seed(args.seed)
    init_distributed_mode(args)

    model = build_memoryvln_inference_from_args(args)

    evaluate(model, args)



def evaluate(model, args):
    world_size = get_world_size()
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        epoch=0,
        args=args,
    )
    evaluator.eval_action(get_rank())

if __name__ == "__main__":
    eval()
