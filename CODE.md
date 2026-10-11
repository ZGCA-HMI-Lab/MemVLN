# MemoryVLN Evaluation Guide

This repository contains the **evaluation (testing) code** for:

**MemoryVLN: From Ego-Aware Memory Representation to Cross-Context Memory Reuse in Vision-Language Navigation**

Training code and training-data construction are intentionally not released.
This package provides the closed-loop evaluation pipeline for R2R-CE and RxR-CE,
together with the model and memory implementation needed to run a trained
MemoryVLN checkpoint.

Project page: <https://zgca-hmi-lab.github.io/MemVLN/>

## Reproducibility scope

This package defines the closed-loop evaluation configuration starting from
standard expert trajectory frames. Exact numerical reproduction additionally
requires the same external R2R-CE/RxR-CE annotations, Matterport3D scenes,
trajectory frames, Qwen and VGGT weights, and trained MemoryVLN checkpoint.
These licensed or large artifacts are not redistributed.

## Method-to-code overview

- Long Sequence Compression (frame selection at inference):
  `src/Memory/memory.py` and `src/evaluation.py`.
- Ego-Aware Visuospatial Representation:
  `src/qwen_vl/model/modeling_qwen2_5_vl.py`,
  `src/qwen_vl/model/position_encoding.py`, and
  `src/qwen_vl/model/vggt/`.
- Hierarchical Memory Bank and Memory Enhancement:
  `src/Memory/memory.py`.
- Inference and action prediction:
  `src/evaluation.py` and `src/qwen_vl/model/`.

The model uses Qwen2.5-VL for language-conditioned action prediction and a
frozen VGGT branch for geometry features. The output action set is
MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, and STOP.

## Reference configuration

| Item | Value |
| --- | --- |
| Evaluation seed | 42 |
| Frame-sampling chunk size | 8 |
| Inner / outer sampling scale | 5.0 / 3.0 |
| Sampling start | History longer than 16 frames |
| Adaptive budget | max(min(N, floor(4 + 2 * sqrt(N))), 16) |
| Forced retention | First and latest observations |
| Agent pose | Relative 6D pose, first observation as reference |
| Position encoding | Enabled |
| Frame memory | 4 short-term + 12 long-term slots |
| Scene-bank capacity | 300 |
| Scene retrieval threshold | 0.6 |
| Scene warm-start coefficient | 0.5 |
| Write-back confidence threshold | 0.4 |
| Maximum evaluation rollout | 400 actions |
| Precision | bfloat16 |
| Geometry backbone | VGGT frozen |

## Directory layout

~~~text
<repository root>/
  LICENSE_APACHE-2.0.txt
  LICENSE_VGGT.txt
  THIRD_PARTY_NOTICES.md
  README.md
  CODE.md
  config/
    vln_r2r.yaml
    vln_rxr.yaml
  scripts/
    evaluation.sh
    evaluation_vit_budget.sh
  src/
    Memory/
    habitat_extensions/
    qwen_vl/
      data/
      model/
      utils/
    utils/
    evaluation.py
  data/                         external datasets and trajectory frames
  models/                       downloaded Qwen and VGGT models
  checkpoints/                  trained MemoryVLN checkpoints
  evaluation/                   evaluation outputs
~~~

All repository defaults use paths relative to the repository root.

## 1. Environment setup

The reference environment uses Python 3.9.23, Habitat-Sim/Habitat-Lab 0.2.4,
PyTorch 2.6.0 with CUDA 12.4, Transformers 4.50.0, and
FlashAttention 2.7.0.post2.

~~~bash
conda create -n memvln python=3.9 -y
conda activate memvln

# Install Habitat-Sim 0.2.4 in headless mode.
conda install habitat-sim==0.2.4 headless -c conda-forge -c aihabitat

# Place the official Habitat-Lab/Baselines v0.2.4 source under habitat-lab/.
pip install -e habitat-lab/habitat-lab
pip install -e habitat-lab/habitat-baselines

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
~~~

Install Habitat-Sim before the pip requirements. FlashAttention must match the
installed CUDA and PyTorch versions; use a compatible prebuilt package when the
cluster does not provide a compiler toolchain.

The evaluation launcher never installs or updates dependencies. Install the
required system libraries once before running it. On Debian or Ubuntu, a typical
headless setup is:

~~~bash
sudo apt-get update
sudo apt-get install -y \
  libegl1-mesa libegl1-mesa-dev libgl1-mesa-dev \
  libgles2-mesa-dev libgl1-mesa-glx libglib2.0-0
~~~

Video export is optional; when it is needed, install an ffmpeg executable
through the system package manager or conda. The `imageio-ffmpeg` package listed
in `requirements.txt` can also provide a local ffmpeg binary.

Verify the core environment before evaluation:

~~~bash
python - <<'PY'
from importlib.metadata import version
import torch

for package in (
    "habitat-sim",
    "habitat-lab",
    "habitat-baselines",
    "transformers",
    "flash-attn",
):
    print(package, version(package))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
PY

nvidia-smi
~~~

For result-level reproduction, do not silently substitute different Habitat,
Transformers, FlashAttention, CUDA, or PyTorch versions.

## 2. Model preparation

Download the Qwen2.5-VL-7B-Instruct model weights and processor files from the
official Qwen model release or its official model-hosting page. Place the full
model directory at:

~~~text
models/Qwen2.5-VL-7B-Instruct/
~~~

Download the VGGT-1B weights and configuration from the official VGGT project
or its official model release page. Place the model directory at:

~~~text
models/VGGT-1B/
~~~

Place a trained MemoryVLN checkpoint at:

~~~text
checkpoints/memoryvln/
~~~

Do not commit downloaded weights or checkpoints. The repository expects local
model files and does not contain authentication tokens or download credentials.

During evaluation, PROCESSOR_PATH must point to the complete Qwen processor
directory even when CHECKPOINT_PATH points to a MemoryVLN checkpoint.

## 3. Data preparation

Place external data under the relative data directory:

~~~text
data/
  scene_datasets/
    mp3d/
      ...
  datasets/
    r2r/
      val_unseen/val_unseen.json.gz
    rxr/
      val_unseen/val_unseen_guide.json.gz
      val_unseen/val_unseen_guide_gt.json.gz
  trajectory_data/
    R2R/val_unseen/<episode_id>/step_*.png
    RxR/val_unseen/<episode_id>/step_*.png
~~~

Evaluation consumes existing standard trajectory frames; it does not collect
simulator rollouts. Frame filenames must contain the corresponding navigation
action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP, using a stable numeric
order such as `step_0000_TURN_RIGHT.png`. Each filename must describe the action
predicted from that observation.

## 4. Closed-loop evaluation

The included Habitat configurations already use relative data paths. Run all
commands from the repository root.

Use a fixed evaluation process count and a new output directory for every
independent run. Existing result files are treated as resumable state and
completed episodes are skipped.

Common evaluation settings:

~~~bash
export WORLD_SIZE=1
export EVAL_SEED=42
export EVAL_MAX_STEPS=400
export EVAL_LIMIT_EPISODES=0
export EVAL_SAVE_VIDEO=0
export EVAL_PROFILE_INFERENCE_SPEED=0

export USE_POSITION_ENCODING=1
export EVAL_MAX_MEMORY_FRAMES=16
export EVAL_MEMORY_SHORT_CAPACITY=4
export EVAL_MEMORY_LONG_CAPACITY=12

export EVAL_MAX_IMAGE_FRAMES=-1
export EVAL_EBBINGHAUS_CHUNK_SIZE=8
export EVAL_EBBINGHAUS_S=5.0
export EVAL_EBBINGHAUS_S_OUTER=3.0
export EVAL_SHRINK_START_FRAMES=16
export EVAL_SHRINK_ADAPTIVE_OFFSET=4
export EVAL_SHRINK_ADAPTIVE_COEF=2
export EVAL_SHRINK_MIN_FRAMES=16

export EVAL_SKIP_FRAME_SAMPLING=0
export EVAL_PHASED_MEMORY=0
export EVAL_VGGT_KV_TRIM_8_48=0
export EVAL_NO_VGGT_KV_TRIM=0
~~~

WORLD_SIZE=1 gives the simplest fixed episode order. If a released result was
produced with distributed evaluation, use the same process count and partition
because the context-level memory evolves with episode order.

### R2R-CE

~~~bash
export CHECKPOINT_PATH=checkpoints/memoryvln
export PROCESSOR_PATH=models/Qwen2.5-VL-7B-Instruct
export CONFIG=config/vln_r2r.yaml
export EVAL_SPLIT=val_unseen
export OUTPUT_PATH=evaluation/r2r_val_unseen
export RESULT_JSON=result_val_unseen.json

bash scripts/evaluation.sh
~~~

### RxR-CE

~~~bash
export CHECKPOINT_PATH=checkpoints/memoryvln
export PROCESSOR_PATH=models/Qwen2.5-VL-7B-Instruct
export CONFIG=config/vln_rxr.yaml
export EVAL_SPLIT=val_unseen
export EVAL_HABITAT_DATA_PATH=data/datasets/rxr/val_unseen/val_unseen_guide.json.gz
export OUTPUT_PATH=evaluation/rxr_val_unseen
export RESULT_JSON=result_val_unseen.json

bash scripts/evaluation.sh
~~~

For RxR nDTW/SDTW, keep val_unseen_guide_gt.json.gz in the same directory as
the selected guide annotation.

Evaluation uses the same frame-selection rule as training. The default memory
contains four short-term slots and twelve long-term slots, and 6D pose encoding
is enabled.

For a complete val-unseen run, verify the final JSON contains the expected
number of unique episodes: 1,839 for R2R-CE and 3,669 for RxR-CE. The final
console summary reports SR, SPL, OS, NE, and, when configured, nDTW and SDTW.
The result JSON stores per-episode metrics so the aggregates can be recomputed.

Use EVAL_LIMIT_EPISODES=2 only for a smoke test. Return it to 0 and choose a new
OUTPUT_PATH before the full run. Partial rank files are written during
evaluation and merged into RESULT_JSON when all ranks finish.

## Artifact and run manifest

For every reported run, record:

- the source revision or submission archive checksum;
- the Qwen, VGGT, and MemoryVLN checkpoint identifiers and checksums;
- the output of pip freeze and the environment verification command;
- GPU model, GPU count, CUDA driver, and CUDA runtime;
- evaluation WORLD_SIZE, split, seed, and output directory;
- the exact evaluation shell commands.

Example commands:

~~~bash
sha256sum checkpoints/memoryvln/config.json
sha256sum checkpoints/memoryvln/*.safetensors
python -m pip freeze > evaluation/pip-freeze.txt
nvidia-smi -q > evaluation/nvidia-smi.txt
~~~

## Reproducibility checklist

Before comparing metrics, confirm all of the following:

1. The software versions match the reference environment.
2. The same external dataset splits and Matterport3D scenes are used.
3. The trajectory frame names, ordering, dimensions, and actions are unchanged.
4. The same Qwen, VGGT, and MemoryVLN checkpoint weights are used.
5. Seed 42 is used for evaluation.
6. Evaluation uses the same WORLD_SIZE and starts from an empty output path.
7. The checkpoint config enables the intended memory, scene-bank, confidence,
   compression, and position-encoding components.
8. Final result files contain the expected episode count and no duplicate
   scene/episode identifiers.

## Running notes

- Activate the Python environment before running the launcher.
- Always run commands from the repository root.
- Keep Qwen processor files with the downloaded Qwen model directory.
- Large-context evaluation is memory intensive.
- The evaluation launcher only checks dependencies. It exits with an actionable
  error when Habitat, Mesa/EGL, or an enabled video-export dependency is missing.
- Do not commit datasets, model weights, checkpoints, caches, videos, logs, or
  evaluation outputs.
- Full deterministic mode is not enabled. Fixed seeds make runs comparable, but
  CUDA kernels, FlashAttention, distributed reduction order, dataloader
  scheduling, and evaluation episode partitioning may prevent bitwise-identical
  outputs.
- A different evaluation WORLD_SIZE is a different experimental configuration.

## Privacy and submission scope

The package uses generic relative paths and contains no API keys, access tokens,
passwords, personal email addresses, personal names, or development machine
paths. External models and datasets must be obtained independently.
