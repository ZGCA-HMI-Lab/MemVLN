#!/usr/bin/env python3
"""
Build MemoryVLN R2R-CE / RxR-CE training annotations with **relative** agent poses.
Each training step contains its full observation history before applying the adaptive
**two-level Ebbinghaus sampling** used by ``src/qwen_vl/data/data_qwen.py``.

Pose: dead-reckoning from ``step_*.png`` actions; ``forward_step_size`` / ``turn_angle``
match ``config/vln_r2r.yaml``.

Output: TSV ``id<TAB>json`` with optional ``agent_pose``, ``step_index``,
``num_history_frames`` for parity with shrunk pose datasets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import glob
import gzip
import json
import math
import os
import re
import sys
from functools import partial

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Memory.memory import two_level_ebbinghaus_indices

_ACTION_RE = re.compile(r"(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)", re.I)

def update_conversation_image_count(value: str, new_K: int) -> str:
    """Update image placeholders after MemoryVLN frame sampling."""
    pat = r"((?:observation memories:|historical observation:|This is your historical observation:))(<image>)*(\n[^\n]*current observation:)(<image>)"

    def repl(m):
        g1, g2, g3, g4 = m.groups()
        return g1 + "<image>" * (new_K - 1) + g3 + "<image>"

    return re.sub(pat, repl, value, count=1)


def compute_shrink_k_target(
    n_frames: int,
    adaptive_offset: float = 4.0,
    adaptive_coef: float = 2.0,
    min_frames: int = 16,
) -> int:
    """K_target = max(min(N, offset + coef*sqrt(N)), min_frames)."""
    adaptive_k = min(n_frames, int(adaptive_offset + adaptive_coef * (n_frames**0.5)))
    return max(adaptive_k, min_frames)


def shrink_sample(
    sample: dict,
    chunk_size: int,
    s_inner: float,
    s_outer: float,
    max_image_frames: int = -1,
    seed: int = 42,
    shrink_start_frames: int = 16,
    shrink_adaptive_offset: float = 4.0,
    shrink_adaptive_coef: float = 2.0,
    shrink_min_frames: int = 16,
) -> dict:
    """
    Apply the adaptive two-level Ebbinghaus policy used by MemoryVLN training and evaluation.

    Default budget: history length N > shrink_start_frames 时抽帧；
    K_target = max(min(N, offset + coef*sqrt(N)), min_frames)，再经双层艾宾浩斯落到 K 帧。
    """
    images = sample.get("images")
    if not images:
        return sample

    N_original = len(images)
    if N_original <= shrink_start_frames:
        return sample

    K_target = compute_shrink_k_target(
        N_original,
        adaptive_offset=shrink_adaptive_offset,
        adaptive_coef=shrink_adaptive_coef,
        min_frames=shrink_min_frames,
    )
    if max_image_frames is not None and max_image_frames > 0:
        K_target = min(K_target, max_image_frames)

    if N_original <= K_target:
        return sample

    rng = np.random.default_rng(seed)
    keep_indices = two_level_ebbinghaus_indices(
        num_frames=N_original,
        max_frames=K_target,
        chunk_size=chunk_size,
        s_inner=s_inner,
        s_outer=s_outer,
        rng=rng,
    )

    sample = dict(sample)
    sample["images"] = [images[j] for j in keep_indices]

    if "agent_poses" in sample and len(sample["agent_poses"]) == N_original:
        sample["agent_poses"] = [sample["agent_poses"][j] for j in keep_indices]

    K = len(keep_indices)
    for conv in sample.get("conversations", []):
        if conv.get("from") == "human" or conv.get("role") == "user":
            content = conv.get("value", conv.get("content", ""))
            conv["value"] = conv.get("value", content)
            conv["value"] = update_conversation_image_count(conv["value"], K)
            if "content" in conv:
                conv["content"] = conv["value"]
            break

    if "num_history_frames" in sample:
        sample["num_history_frames"] = K - 1

    return sample


def parse_action_from_stem(stem: str) -> str:
    m = _ACTION_RE.findall(stem.upper())
    if not m:
        raise ValueError(f"Cannot parse action from filename stem: {stem!r}")
    return m[-1].upper()


def sorted_image_paths(img_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(img_dir, "*.png"))
    if not paths:
        paths = glob.glob(os.path.join(img_dir, "*.jpg"))

    def _key(p: str) -> tuple[int, str]:
        b = os.path.basename(p)
        m = re.search(r"step_(\d+)", b)
        return (int(m.group(1)) if m else 0, b)

    return sorted(paths, key=_key)


def _pose_dict_planar(x: float, z: float, yaw: float) -> dict:
    """Relative planar pose; euler [0, yaw, 0]; quaternion [w,x,y,z] for rotation about Y."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return {
        "position": [float(x), 0.0, float(z)],
        "rotation_euler": [0.0, float(yaw), 0.0],
        "rotation_quaternion": [float(cy), 0.0, float(sy), 0.0],
    }


def integrate_relative_poses(
    actions_per_frame: list[str],
    forward_step: float,
    turn_angle_rad: float,
) -> list[dict]:
    x = z = 0.0
    yaw = 0.0
    out: list[dict] = []

    for i in range(len(actions_per_frame)):
        out.append(_pose_dict_planar(x, z, yaw))
        if i == len(actions_per_frame) - 1:
            break
        a = actions_per_frame[i]
        if a == "STOP":
            continue
        if a == "TURN_LEFT":
            yaw += turn_angle_rad
        elif a == "TURN_RIGHT":
            yaw -= turn_angle_rad
        elif a == "MOVE_FORWARD":
            x += forward_step * math.sin(yaw)
            z += forward_step * math.cos(yaw)
        else:
            raise ValueError(f"Unknown action: {a}")

    return out


def _episode_instruction(ep: dict) -> str:
    inst = ep["instruction"]
    if isinstance(inst, dict):
        return str(inst["instruction_text"]).strip()
    return str(inst).strip()


def process_episode_vlnce_with_poses(
    ep: dict,
    img_root: str,
    forward_step: float,
    turn_angle_rad: float,
) -> list[dict]:
    episode_id = str(ep["episode_id"])
    instruction = _episode_instruction(ep)
    img_dir = os.path.join(img_root, episode_id)

    img_files = sorted_image_paths(img_dir)
    if not img_files:
        return []

    stems = [os.path.splitext(os.path.basename(p))[0] for p in img_files]
    try:
        actions = [parse_action_from_stem(s) for s in stems]
    except ValueError:
        return []

    poses = integrate_relative_poses(actions, forward_step, turn_angle_rad)

    episode_results: list[dict] = []
    num_images = len(img_files)

    for i in range(num_images):
        # MemoryVLN uses the complete history [0..i] before adaptive frame sampling.
        idxs = list(range(i + 1))
        sampled_imgs = [img_files[j] for j in idxs]
        sampled_imgs = [p for p in sampled_imgs if os.path.exists(p)]
        if not sampled_imgs:
            continue

        his_img_tags = "<image>" * (len(sampled_imgs) - 1)

        if i == num_images - 1:
            action = "STOP"
        else:
            name_parts = os.path.basename(img_files[i]).replace(".png", "").split("_")
            if len(name_parts) > 3:
                action = f"{name_parts[-2].upper()}_{name_parts[-1].upper()}"
            else:
                action = name_parts[-1].upper() if name_parts else "MOVE_FORWARD"

        conversations = [
            {
                "from": "human",
                "value": (
                    "You are a visual language navigation model, and your should go to the "
                    "locations to complete the given task. Compare the observation and "
                    "instruction to infer your current progress, and then select the correct "
                    "direction from the candidates to go to the target location and finish the "
                    "task.\n"
                    f" This is your historical observation:{his_img_tags}\n"
                    " This is your current observation:<image>\n"
                    f" Your task is to {instruction}\n"
                    " You should take one of the following actions:\n"
                    " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
                ),
            },
            {"from": "gpt", "value": action},
        ]

        pose_for_sample = []
        for j in idxs:
            if j < len(poses):
                pose_for_sample.append(copy.deepcopy(poses[j]))
            else:
                pose_for_sample.append(_pose_dict_planar(0.0, 0.0, 0.0))

        if len(pose_for_sample) != len(sampled_imgs):
            continue

        pose_i = copy.deepcopy(poses[i]) if i < len(poses) else _pose_dict_planar(0, 0, 0)

        sample_dict = {
            "id": f"{episode_id}/{os.path.basename(img_files[i])}",
            "conversations": conversations,
            "images": sampled_imgs,
            "agent_pose": pose_i,
            "agent_poses": pose_for_sample,
            "step_index": i,
            "num_history_frames": len(sampled_imgs) - 1,
        }
        episode_results.append(sample_dict)

    return episode_results


def main():
    DEFAULT_OUT_MAIN = "train_r2r_rxr_relative_pose_shrunk.tsv"

    parser = argparse.ArgumentParser(
        description="MemoryVLN VLN-CE relative poses: full history + adaptive two-level Ebbinghaus sampling."
    )
    parser.add_argument("--r2r_img_root", default="data/trajectory_data/R2R/train")
    parser.add_argument("--r2r_json_gz", default="data/datasets/r2r/train/train.json.gz")
    parser.add_argument("--rxr_img_root", default="data/trajectory_data/RxR/train")
    parser.add_argument("--rxr_json_gz", default="data/datasets/rxr/train/train_guide.json.gz")
    parser.add_argument("--forward_step_size", type=float, default=0.25)
    parser.add_argument("--turn_angle_deg", type=float, default=15.0)
    parser.add_argument("--max_episodes", type=int, default=-1)
    parser.add_argument("--skip_rxr", action="store_true")
    parser.add_argument("--skip_r2r", action="store_true")
    parser.add_argument("--format", choices=("tsv", "json"), default="tsv")
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUT_MAIN,
        help="Output path for the standard R2R/RxR relative-pose TSV.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--no_shrink",
        action="store_true",
        help="Keep full-history rows (no Ebbinghaus pass); huge files.",
    )
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--ebbinghaus_s", type=float, default=5.0)
    parser.add_argument("--ebbinghaus_s_outer", type=float, default=3.0)
    parser.add_argument(
        "--shrink_start_frames",
        type=int,
        default=16,
        help="历史帧数 N<=该值时不抽帧（与 MemoryVLN 训练和评测策略一致）",
    )
    parser.add_argument(
        "--shrink_adaptive_offset",
        type=float,
        default=4.0,
        help="K_target 公式常数项：offset + coef*sqrt(N)",
    )
    parser.add_argument(
        "--shrink_adaptive_coef",
        type=float,
        default=2.0,
        help="K_target 公式 sqrt 系数",
    )
    parser.add_argument(
        "--shrink_min_frames",
        type=int,
        default=16,
        help="K_target 下限 max(..., min_frames)",
    )
    parser.add_argument("--max_image_frames", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    turn_rad = math.radians(args.turn_angle_deg)
    all_results: list[dict] = []

    worker_vlnce = partial(
        process_episode_vlnce_with_poses,
        forward_step=args.forward_step_size,
        turn_angle_rad=turn_rad,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        if not args.skip_r2r:
            with gzip.open(args.r2r_json_gz, "rt", encoding="utf-8") as f:
                data_r2r = json.load(f)
            eps = data_r2r["episodes"]
            if args.max_episodes > 0:
                eps = eps[: args.max_episodes]
            fn = partial(worker_vlnce, img_root=args.r2r_img_root)
            for episode_res in tqdm(ex.map(fn, eps, chunksize=4), total=len(eps), desc="R2R"):
                all_results.extend(episode_res)

        if not args.skip_rxr:
            with gzip.open(args.rxr_json_gz, "rt", encoding="utf-8") as f:
                data_rxr = json.load(f)
            eps = data_rxr["episodes"]
            if args.max_episodes > 0:
                eps = eps[: args.max_episodes]
            fn = partial(worker_vlnce, img_root=args.rxr_img_root)
            for episode_res in tqdm(ex.map(fn, eps, chunksize=4), total=len(eps), desc="RxR"):
                all_results.extend(episode_res)

    if not all_results:
        print(
            "[error] 未生成任何样本；请检查 R2R/RxR 图像与 json.gz。",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.no_shrink:
        print(
            "[shrink] "
            f"start>{args.shrink_start_frames}, "
            f"K_target=max(min(N, {args.shrink_adaptive_offset:g}"
            f"+{args.shrink_adaptive_coef:g}*sqrt(N)), {args.shrink_min_frames}), "
            f"chunk={args.chunk_size}, s_inner={args.ebbinghaus_s}, "
            f"s_outer={args.ebbinghaus_s_outer}",
            flush=True,
        )
        shrunk_rows: list[dict] = []
        for idx, row in enumerate(
            tqdm(all_results, desc="shrink (Ebbinghaus)", mininterval=2.0)
        ):
            shrunk_rows.append(
                shrink_sample(
                    row,
                    chunk_size=args.chunk_size,
                    s_inner=args.ebbinghaus_s,
                    s_outer=args.ebbinghaus_s_outer,
                    max_image_frames=args.max_image_frames,
                    seed=args.seed + idx,
                    shrink_start_frames=args.shrink_start_frames,
                    shrink_adaptive_offset=args.shrink_adaptive_offset,
                    shrink_adaptive_coef=args.shrink_adaptive_coef,
                    shrink_min_frames=args.shrink_min_frames,
                )
            )
        all_results = shrunk_rows

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if args.format == "json":
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for row in all_results:
                key = row["id"].replace("\t", " ")
                f.write(key + "\t" + json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_results)} samples to {args.output}")


if __name__ == "__main__":
    main()
