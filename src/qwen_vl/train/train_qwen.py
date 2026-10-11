# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwen_vl.train.lora_utils import (
    apply_lora_to_llm,
    load_lora_adapter_if_present,
    merge_lora_into_base,
    print_memoryvln_trainable_summary,
    save_lora_adapter,
    resolve_lora_base_and_adapter,
    write_lora_training_meta,
)
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLImageProcessor,
    Trainer,
    TrainerCallback,
    AutoConfig,
    set_seed,
    enable_full_determinism,
)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def _resolve_processor_path(model_path: str, vggt_model_path: str) -> str:
    """MemoryVLN/DeepSpeed checkpoint 常无 preprocessor_config.json，回退到底座 Qwen2.5-VL。"""
    if os.path.isfile(os.path.join(model_path, "preprocessor_config.json")):
        return model_path

    explicit = os.environ.get("PROCESSOR_PATH", "").strip()
    if explicit and os.path.isfile(os.path.join(explicit, "preprocessor_config.json")):
        rank0_print(f"[processor] 使用 PROCESSOR_PATH={explicit}")
        return explicit

    model_root = os.path.dirname(vggt_model_path.rstrip("/"))
    for sub in ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-3B-Instruct"):
        cand = os.path.join(model_root, sub)
        if os.path.isfile(os.path.join(cand, "preprocessor_config.json")):
            rank0_print(
                f"[processor] checkpoint 无 preprocessor_config.json，回退到 {cand}"
            )
            return cand

    raise FileNotFoundError(
        f"未在 {model_path} 找到 preprocessor_config.json，且未找到 PROCESSOR_PATH "
        f"或 {model_root}/Qwen2.5-VL-*-Instruct；请设置 PROCESSOR_PATH=models/Qwen2.5-VL-7B-Instruct"
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def configure_action_token_lm_head(model_args, model, tokenizer):
    if not getattr(model_args, "train_action_token_rows", False):
        return []
    if model_args.tune_mm_llm or model_args.tune_lm_head or model_args.use_lora:
        raise ValueError(
            "train_action_token_rows cannot be combined with full LLM, full lm_head, or LoRA training"
        )

    action_texts = [
        text.strip()
        for text in str(model_args.action_token_texts).split(",")
        if text.strip()
    ]
    if not action_texts:
        raise ValueError("action_token_texts must contain at least one action")

    token_ids = set()
    for action in action_texts:
        for variant in (action, " " + action):
            token_ids.update(tokenizer.encode(variant, add_special_tokens=False))
    token_ids = sorted(int(token_id) for token_id in token_ids)

    weight = model.lm_head.weight
    weight.requires_grad = True
    row_mask = torch.zeros(weight.shape[0], dtype=torch.float32)
    row_mask[token_ids] = 1.0
    model.register_buffer("action_token_row_mask", row_mask, persistent=False)

    def mask_non_action_rows(grad):
        mask = model.action_token_row_mask.to(device=grad.device, dtype=grad.dtype)
        return grad * mask.unsqueeze(1)

    model._action_token_gradient_hook = weight.register_hook(mask_non_action_rows)
    model.config.train_action_token_rows = True
    model.config.action_token_ids = token_ids
    rank0_print(
        f"[partial-ft] trainable action lm_head rows={token_ids}; "
        f"tokens={[tokenizer.decode([token_id]) for token_id in token_ids]}"
    )
    return token_ids


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    for _, p in model.model.named_parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False

    last_n = int(getattr(model_args, "unfreeze_llm_last_n_layers", 0))
    model.config.unfreeze_llm_last_n_layers = last_n
    if last_n < 0:
        raise ValueError("unfreeze_llm_last_n_layers must be >= 0")
    if model_args.tune_mm_llm and last_n > 0:
        raise ValueError(
            "tune_mm_llm=True conflicts with unfreeze_llm_last_n_layers > 0"
        )

    if model_args.tune_mm_llm:
        for _, p in model.model.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif last_n > 0:
        layers = getattr(model.model, "layers", None)
        if layers is None:
            raise AttributeError("model.model does not expose decoder layers")
        if last_n > len(layers):
            raise ValueError(
                f"Cannot unfreeze last {last_n} layers; model only has {len(layers)}"
            )
        for layer in layers[-last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    if model_args.tune_lm_head and not model_args.tune_mm_llm:
        for p in model.lm_head.parameters():
            p.requires_grad = True

    # vggt is frozen
    for n, p in model.vggt.named_parameters():
        p.requires_grad = False
    for n, p in model.merger.named_parameters():
        p.requires_grad = True

    _lam_mem = getattr(model, "lam_mem", None)
    if _lam_mem is not None:
        _lam_mem.requires_grad = (
            model_args.tune_mm_mlp
            or model_args.tune_mm_llm
            or last_n > 0
        )


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    requested_model_path = model_args.model_name_or_path
    model_load_path, lora_adapter_path = resolve_lora_base_and_adapter(requested_model_path)
    if lora_adapter_path is not None:
        rank0_print(
            f"[train] LoRA checkpoint detected: load base={model_load_path}; "
            f"adapter={lora_adapter_path}"
        )

    config = AutoConfig.from_pretrained(requested_model_path)
    _archs = getattr(config, "architectures", None) or []
    _use_qwen25_memoryvln = (
        getattr(config, "model_type", "") == "qwen2_5_vl"
        or "qwen2.5" in model_args.model_name_or_path.lower()
        or any("Qwen2_5" in str(a) or "MemoryVLN" in str(a) for a in _archs)
    )

    if _use_qwen25_memoryvln:
        from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationForMemoryVLN
        setattr(config, "lam", model_args.lam)
        setattr(config, "reference_frame", model_args.reference_frame)
        setattr(config, "use_position_encoding", model_args.use_position_encoding)
        setattr(config, "use_memory_bank", model_args.use_memory_bank)
        if model_args.lam_mem is not None:
            setattr(config, "lam_mem", model_args.lam_mem)
        setattr(config, "use_memory_compression", model_args.use_memory_compression)
        setattr(config, "max_memory_frames", model_args.max_memory_frames)
        setattr(config, "memory_short_capacity", model_args.memory_short_capacity)
        setattr(config, "memory_long_capacity", model_args.memory_long_capacity)
        setattr(config, "use_scene_memory_bank", model_args.use_scene_memory_bank)
        setattr(config, "scene_bank_capacity", model_args.scene_bank_capacity)
        setattr(config, "scene_retrieval_threshold", model_args.scene_retrieval_threshold)
        setattr(config, "scene_warm_tau", model_args.scene_warm_tau)
        setattr(config, "scene_top_k", model_args.scene_top_k)
        setattr(config, "use_scene_init_loss", model_args.use_scene_init_loss)
        setattr(config, "scene_init_loss_weight", model_args.scene_init_loss_weight)
        setattr(config, "scene_bank_ddp_sync", model_args.scene_bank_ddp_sync)
        setattr(config, "use_frame_confidence", model_args.use_frame_confidence)
        setattr(config, "memory_conf_tau", model_args.memory_conf_tau)
        setattr(config, "memory_conf_max", model_args.memory_conf_max)
        setattr(config, "memory_writeback_conf_threshold", model_args.memory_writeback_conf_threshold)
        setattr(config, "scene_key_lang_weight", model_args.scene_key_lang_weight)
        setattr(config, "scene_use_instruction_key", model_args.scene_use_instruction_key)
        if model_args.vit_pose_mode is not None:
            setattr(config.vision_config, "vit_pose_mode", model_args.vit_pose_mode)
        model = Qwen2_5_VLForConditionalGenerationForMemoryVLN.from_pretrained(
            pretrained_model_name_or_path=model_load_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            vggt_model_path=model_args.vggt_model_path
        )
        _lam_mem_param = getattr(model, "lam_mem", None)
        if _lam_mem_param is not None:
            model.config.lam_mem = float(_lam_mem_param.detach().cpu().item())

        _processor_path = _resolve_processor_path(
            model_args.model_name_or_path, model_args.vggt_model_path
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            _processor_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_load_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if model_args.use_lora and model_args.tune_mm_llm:
        rank0_print(
            "use_lora=True: tune_mm_llm is ignored; base LLM frozen with LoRA adapters."
        )
        model_args.tune_mm_llm = False
    if model_args.use_lora and model_args.unfreeze_llm_last_n_layers > 0:
        raise ValueError(
            "use_lora cannot be combined with unfreeze_llm_last_n_layers > 0"
        )

    set_model(model_args, model)
    configure_action_token_lm_head(model_args, model, tokenizer)
    if model_args.unfreeze_llm_last_n_layers > 0:
        trainable_layers = [
            idx
            for idx, layer in enumerate(model.model.layers)
            if any(p.requires_grad for p in layer.parameters())
        ]
        rank0_print(
            f"[partial-ft] trainable LLM layers={trainable_layers}; "
            f"lm_head_trainable={any(p.requires_grad for p in model.lm_head.parameters())}"
        )

    if model_args.use_lora:
        if lora_adapter_path is not None:
            model = load_lora_adapter_if_present(
                model, requested_model_path, is_trainable=True
            )
            if getattr(model_args, "lora_train_lm_head", True):
                if hasattr(model, "lm_head") and model.lm_head is not None:
                    for p in model.lm_head.parameters():
                        p.requires_grad = True
        else:
            model = apply_lora_to_llm(model, model_args)
        if training_args.gradient_checkpointing and hasattr(
            model.model, "enable_input_require_grads"
        ):
            model.model.enable_input_require_grads()

    if torch.distributed.get_rank() == 0:
        if model_args.use_lora:
            print_memoryvln_trainable_summary(model)
        else:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    callbacks = []
    if model_args.use_lora:

        class LoraCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, model=None, **kwargs):
                if not state.is_world_process_zero or model is None:
                    return
                ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                save_lora_adapter(model, ckpt)

        callbacks.append(LoraCheckpointCallback())

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    resume_setting = os.environ.get("RESUME_FROM_CHECKPOINT", "auto").strip()
    resume_lower = resume_setting.lower()
    if resume_lower in ("0", "false", "no", "off", "none"):
        rank0_print("[train] automatic checkpoint resume disabled")
        trainer.train()
    elif resume_lower == "auto":
        if checkpoint_dirs:
            rank0_print("[train] checkpoint found; resuming latest checkpoint")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    else:
        rank0_print(f"[train] resuming checkpoint: {resume_setting}")
        trainer.train(resume_from_checkpoint=resume_setting)
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    _chat_src = os.path.join(model_args.model_name_or_path, "chat_template.json")
    if not os.path.isfile(_chat_src) and _use_qwen25_memoryvln:
        _chat_src = os.path.join(
            _resolve_processor_path(model_args.model_name_or_path, model_args.vggt_model_path),
            "chat_template.json",
        )
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    if os.path.isfile(_chat_src):
        shutil.copy2(_chat_src, template_path)

    model.config.use_cache = True

    if model_args.use_lora:
        if model_args.lora_merge_before_save:
            merge_lora_into_base(trainer.model)
        else:
            if torch.distributed.get_rank() == 0:
                adapter_dir = save_lora_adapter(trainer.model, training_args.output_dir)
                write_lora_training_meta(
                    training_args.output_dir,
                    model_args,
                    model_load_path,
                )
                if adapter_dir:
                    rank0_print(f"LoRA adapter saved to {adapter_dir}")

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
