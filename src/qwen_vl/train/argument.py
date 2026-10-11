import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="models/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=False)
    unfreeze_llm_last_n_layers: int = field(
        default=0,
        metadata={
            "help": "Directly fine-tune only the last N LLM decoder layers. "
            "Cannot be combined with tune_mm_llm or use_lora."
        },
    )
    tune_lm_head: bool = field(
        default=False,
        metadata={"help": "Train the full vocabulary lm_head outside full-LLM fine-tuning."},
    )
    train_action_token_rows: bool = field(
        default=False,
        metadata={
            "help": "Train only lm_head rows used by navigation action strings. "
            "All other vocabulary rows receive zero gradient."
        },
    )
    action_token_texts: str = field(
        default="STOP,MOVE_FORWARD,TURN_LEFT,TURN_RIGHT",
        metadata={
            "help": "Comma-separated action strings whose bare and leading-space token rows are trainable."
        },
    )
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    vggt_model_path: str = field(default="models/VGGT-1B")
    lam: float = field(default=0.2)
    distill_loss_weight: float = field(default=1.0)
    reference_frame: str = field(default="last")
    use_position_encoding: bool = field(
        default=True,
        metadata={
            "help": "If True, inject agent pose into VGGT features and ViT (see vision_config.vit_pose_mode / use_vit_sine_pose)."
        },
    )
    vit_pose_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override vision injection: 'sine' for fixed 6D sinusoidal pose encoding or "
            "'none' to disable it. If None, uses vision_config "
            "(legacy: use_vit_sine_pose True->sine, False->none)."
        },
    )
    use_memory_bank: bool = field(
        default=False,
        metadata={
            "help": "Enable MemoryVLN FusedTokenMemory after 2D+3D fusion."
        },
    )
    lam_mem: Optional[float] = field(
        default=None,
        metadata={
            "help": "Residual weight for memory-enhanced visual tokens (trainable nn.Parameter). "
            "If omitted, keep value from checkpoint config/weights (do not force 0)."
        },
    )
    use_memory_compression: bool = field(
        default=True,
        metadata={
            "help": "If True (and use_memory_bank), run compress_to_memory_tokens and shrink image_pad; "
            "if False, only chunk-wise memory fusion + lam_mem, no frame compression."
        },
    )
    max_memory_frames: int = field(
        default=16,
        metadata={"help": "FusedTokenMemory total frame slots (4 short + 12 long); also compress_to_memory_tokens cap."},
    )
    memory_short_capacity: int = field(
        default=4,
        metadata={"help": "Recent frames kept verbatim in the frame memory bank."},
    )
    memory_long_capacity: int = field(
        default=12,
        metadata={"help": "Summary slots in the frame memory bank."},
    )
    use_scene_memory_bank: bool = field(
        default=False,
        metadata={"help": "Enable cross-episode scene bank retrieval/write-back."},
    )
    scene_bank_capacity: int = field(
        default=300,
        metadata={"help": "Max entries in in-memory scene bank."},
    )
    scene_retrieval_threshold: float = field(
        default=0.6,
        metadata={"help": "Cosine similarity threshold for scene retrieval hit."},
    )
    scene_warm_tau: float = field(
        default=0.5,
        metadata={"help": "Blend ratio for warm start from retrieved scene prior."},
    )
    scene_top_k: int = field(
        default=1,
        metadata={"help": "Top-k retrieval entries reserved for scene bank (v1 uses top-1)."},
    )
    use_scene_init_loss: bool = field(
        default=False,
        metadata={"help": "Enable optional init alignment loss after warm start."},
    )
    scene_init_loss_weight: float = field(
        default=0.05,
        metadata={"help": "Weight for optional scene init alignment loss."},
    )
    scene_bank_ddp_sync: bool = field(
        default=False,
        metadata={
            "help": "After successful scene write-back, all_gather_object merge scene_bank across DDP ranks."
        },
    )
    use_frame_confidence: bool = field(
        default=False,
        metadata={"help": "SmeC-lite: v_t from frame cosine drift; scale fuse update_ratio and optional write-back gate."},
    )
    memory_conf_tau: float = field(
        default=5.0,
        metadata={"help": "Confidence gate: together with memory_conf_max scales read fusion strength."},
    )
    memory_conf_max: float = field(
        default=10.0,
        metadata={"help": "Upper clamp scale for accumulated per-frame confidence."},
    )
    memory_writeback_conf_threshold: float = field(
        default=0.4,
        metadata={"help": "Skip scene bank write-back when mean short/long confidence is below this."},
    )
    scene_key_lang_weight: float = field(
        default=0.3,
        metadata={"help": "Blend weight for instruction text embedding in scene retrieval/write-back key."},
    )
    scene_use_instruction_key: bool = field(
        default=False,
        metadata={"help": "Mean-pool non-image text token embeddings into scene key and episode text_key cache."},
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Apply PEFT LoRA to the LLM backbone (model.model). Base LLM weights stay frozen; "
            "merger / lam_mem still follow tune_mm_mlp. Much cheaper than full LLM fine-tuning."
        },
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling)."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated LoRA target module names. Default: q/k/v/o_proj + gate/up/down_proj."
        },
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias mode: none, all, or lora_only."},
    )
    lora_train_lm_head: bool = field(
        default=True,
        metadata={"help": "Also train lm_head when use_lora (recommended for action-token prediction)."},
    )
    lora_merge_before_save: bool = field(
        default=False,
        metadata={
            "help": "If True, merge LoRA into base weights before final save (larger checkpoint; "
            "inference without PEFT). Default saves adapter under lora_adapter/ instead."
        },
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(
        default=-1,
        metadata={"help": "Optional hard cap after adaptive frame sampling; -1 keeps the adaptive budget."},
    )
    video_min_frames: Optional[int] = field(default=4)
    ebbinghaus_chunk_size: int = field(default=8)
    ebbinghaus_s: float = field(default=5.0)
    ebbinghaus_s_outer: float = field(default=3.0)
    shrink_start_frames: int = field(default=16)
    shrink_adaptive_offset: float = field(default=4.0)
    shrink_adaptive_coef: float = field(default=2.0)
    shrink_min_frames: int = field(default=16)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
