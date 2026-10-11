"""LoRA helpers for MemoryVLN / Qwen2.5-VL training and inference."""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Union

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ModuleNotFoundError:
    LoraConfig = PeftModel = TaskType = get_peft_model = None

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_lora_target_modules(value: Optional[str]) -> List[str]:
    if value is None or not str(value).strip():
        return list(DEFAULT_LORA_TARGET_MODULES)
    return [m.strip() for m in str(value).split(",") if m.strip()]


def _require_peft() -> None:
    if PeftModel is None:
        raise ModuleNotFoundError("peft is required for LoRA training/inference")


def build_lora_config(model_args) -> LoraConfig:
    _require_peft()
    return LoraConfig(
        r=int(model_args.lora_r),
        lora_alpha=int(model_args.lora_alpha),
        lora_dropout=float(model_args.lora_dropout),
        target_modules=parse_lora_target_modules(model_args.lora_target_modules),
        bias=model_args.lora_bias,
        # Backbone is Qwen2_5_VLModel; loss is computed on the outer MemoryVLN head.
        # CAUSAL_LM would always pass labels=... into the inner model and break forward.
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def apply_lora_to_llm(memoryvln_model, model_args):
    """Wrap ``memoryvln_model.model`` (Qwen2_5_VLModel) with PEFT LoRA adapters."""
    if getattr(model_args, "use_lora", False) is not True:
        return memoryvln_model

    lora_config = build_lora_config(model_args)
    memoryvln_model.model = get_peft_model(memoryvln_model.model, lora_config)

    if getattr(model_args, "lora_train_lm_head", True):
        if hasattr(memoryvln_model, "lm_head") and memoryvln_model.lm_head is not None:
            for p in memoryvln_model.lm_head.parameters():
                p.requires_grad = True

    return memoryvln_model


def is_peft_wrapped_llm(memoryvln_model) -> bool:
    return PeftModel is not None and isinstance(getattr(memoryvln_model, "model", None), PeftModel)


def lora_adapter_dir(pretrained: str) -> Optional[str]:
    """Return directory containing ``adapter_config.json``, if any."""
    if os.path.isfile(os.path.join(pretrained, "adapter_config.json")):
        return pretrained
    nested = os.path.join(pretrained, "lora_adapter")
    if os.path.isfile(os.path.join(nested, "adapter_config.json")):
        return nested
    return None


def resolve_lora_base_and_adapter(pretrained: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a checkpoint path for inference.

    LoRA training checkpoints may contain PEFT-wrapped shard keys in the outer
    directory plus a nested lora_adapter/. In that case the base model must
    be loaded from adapter_config.json and the adapter attached afterward.
    """
    adapter_dir = lora_adapter_dir(pretrained)
    if adapter_dir is None:
        return pretrained, None

    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    try:
        with open(adapter_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return pretrained, adapter_dir

    base_model = cfg.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model.strip():
        return base_model, adapter_dir
    return pretrained, adapter_dir


def load_lora_adapter_if_present(memoryvln_model, pretrained: str, is_trainable: bool = False):
    """Attach LoRA adapter to an already-loaded MemoryVLN model."""
    _require_peft()
    adapter_dir = lora_adapter_dir(pretrained)
    if adapter_dir is None:
        return memoryvln_model

    memoryvln_model.model = PeftModel.from_pretrained(
        memoryvln_model.model,
        adapter_dir,
        is_trainable=is_trainable,
    )
    return memoryvln_model


def save_lora_adapter(memoryvln_model, output_dir: str) -> Optional[str]:
    """Save PEFT adapter weights to ``output_dir/lora_adapter``."""
    if not is_peft_wrapped_llm(memoryvln_model):
        return None

    adapter_dir = os.path.join(output_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    memoryvln_model.model.save_pretrained(adapter_dir)
    return adapter_dir


def write_lora_training_meta(output_dir: str, model_args, base_model_path: str) -> None:
    meta = {
        "base_model_path": base_model_path,
        "use_lora": True,
        "lora_r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "lora_target_modules": parse_lora_target_modules(model_args.lora_target_modules),
        "lora_bias": model_args.lora_bias,
        "lora_train_lm_head": model_args.lora_train_lm_head,
    }
    with open(os.path.join(output_dir, "lora_training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def count_trainable_parameters(module) -> tuple:
    trainable = 0
    total = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def print_memoryvln_trainable_summary(memoryvln_model, tag: str = "MemoryVLN") -> None:
    parts = [
        ("visual", getattr(memoryvln_model, "visual", None)),
        ("llm", getattr(memoryvln_model, "model", None)),
        ("lm_head", getattr(memoryvln_model, "lm_head", None)),
        ("vggt_merger", getattr(memoryvln_model, "merger", None)),
        ("lam_mem", getattr(memoryvln_model, "lam_mem", None)),
    ]
    print(f"[{tag}] trainable parameter summary:")
    grand_trainable = 0
    grand_total = 0
    for name, mod in parts:
        if mod is None:
            continue
        if name == "lam_mem" and hasattr(mod, "numel"):
            t, tot = (1 if mod.requires_grad else 0), 1
            print(f"  {name}: trainable={t:,} total={tot:,}")
            grand_trainable += t
            grand_total += tot
            continue
        t, tot = count_trainable_parameters(mod)
        pct = 100.0 * t / tot if tot else 0.0
        print(f"  {name}: trainable={t:,} total={tot:,} ({pct:.2f}%)")
        grand_trainable += t
        grand_total += tot
    pct = 100.0 * grand_trainable / grand_total if grand_total else 0.0
    print(
        f"[{tag}] total trainable={grand_trainable:,} / {grand_total:,} ({pct:.2f}%)"
    )
    if is_peft_wrapped_llm(memoryvln_model):
        try:
            memoryvln_model.model.print_trainable_parameters()
        except Exception:
            pass


def merge_lora_into_base(memoryvln_model):
    """Merge LoRA into base LLM weights (destructive; for export-only)."""
    if not is_peft_wrapped_llm(memoryvln_model):
        return memoryvln_model
    memoryvln_model.model = memoryvln_model.model.merge_and_unload()
    return memoryvln_model
