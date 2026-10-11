"""
Memory modules and frame-sampling utilities for MemoryVLN.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
from typing import Tuple, Optional, List, Dict, Any, Sequence


# 推理专用：Habitat 等仅设置 fused_memory.inference_compress_max_frames 时，
# compress_to_memory_tokens 使用该上限；训练路径不设置，仍由 config 传入的 max_frames 决定。
INFERENCE_MEMORY_COMPRESS_MAX_FRAMES = 56

# 推理专用：帧级记忆池短/长期槽位数（合计 32）；由 configure_inference_memory_pool 应用，训练不调用。
INFERENCE_MEMORY_SHORT_FRAMES = 8
INFERENCE_MEMORY_LONG_FRAMES = 24
INFERENCE_MEMORY_POOL_MAX_FRAMES = (
    INFERENCE_MEMORY_SHORT_FRAMES + INFERENCE_MEMORY_LONG_FRAMES
)


# ========== Fused Token Memory (for 2D+3D fused visual tokens) ==========

class FusedTokenMemory:
    """
    显式记忆：2D+3D 融合后的视觉 token，按帧维护短/长期记忆池并用余弦 top-k 加权融合。
    读融合统一走 ``fuse_with_memory_per_frame``，并从短期/长期帧池检索。

    Args:
        dimension: token 的维度（通常是 config.hidden_size）
        top_k: ``fuse_with_memory_per_frame`` 中从记忆库取最相似的前 k 条（帧级）
        update_ratio: 融合时记忆特征的权重（0.2 = 20% 记忆，80% 当前）
        temperature: softmax 温度参数（用于加权平均）
        max_frames: 帧级记忆池大小（流式 chunk 模式，默认 16）
        merge_top_k: 仅用于长期池已满时：在多少个最相似槽位上做合并（与 top_k 不同阶段，见下）

        长期池合并时的混合系数在代码内按 age/similarity 自适应计算，不再使用固定 merge_alpha。

        推理时可设实例属性 inference_compress_max_frames（如 56），仅影响 compress_to_memory_tokens
        的合并上限；或调用 configure_inference_memory_pool() 增大短/长期池总容量。训练路径不使用二者。
    """
    
    def __init__(
        self,
        dimension: int = 2048,
        top_k: int = 16,
        update_ratio: float = 0.2,
        temperature: float = 0.2,
        max_frames: int = 16,
        short_capacity: Optional[int] = None,
        long_capacity: Optional[int] = None,
        merge_top_k: int = 8,
        scene_bank_capacity: int = 300,
        scene_retrieval_threshold: float = 0.6,
        scene_warm_tau: float = 0.5,
        scene_top_k: int = 1,
        use_frame_confidence: bool = False,
        memory_conf_tau: float = 5.0,
        memory_conf_max: float = 10.0,
        memory_writeback_conf_threshold: float = 0.4,
        scene_key_lang_weight: float = 0.3,
        scene_bank_ddp_sync: bool = False,
    ):
        self.dimension = dimension
        self.top_k = top_k
        self.update_ratio = update_ratio
        self.temperature = temperature
        # ===== 帧级记忆：短期 + 长期 =====
        # 总帧级容量
        self.max_frames = max_frames
        self.merge_top_k = merge_top_k
        # 短期池容量（最近若干帧原样保留），长期池容量（summary）
        if short_capacity is not None and long_capacity is not None:
            self.short_capacity = short_capacity
            self.long_capacity = long_capacity
        else:
            self.short_capacity = max(1, min(4, max_frames // 4))  # 至少 1，最多占 1/4
            self.long_capacity = max(1, max_frames - self.short_capacity)
        # 短期池：按时间顺序存最近若干帧（完整 patch；内部存 [N_short, P, D]）
        self.short_memory = None  # [N_short, P, D]
        # 长期池：存 summary 帧（完整 patch；内部存 [N_long, P, D]），并维护 age / usage
        self.long_memory = None   # [N_long, P, D]
        self.long_age = None      # [N_long]
        self.long_usage = None    # [N_long]

        # 记录每帧 patch 数（P），仅在首次写入时确定；后续若出现不同的 P，会跳过 patch 级写入
        self.patch_len: Optional[int] = None

        # ===== 跨场景记忆库（进程内） =====
        self.scene_bank_capacity = max(1, int(scene_bank_capacity))
        self.scene_retrieval_threshold = float(scene_retrieval_threshold)
        self.scene_warm_tau = float(scene_warm_tau)
        self.scene_top_k = max(1, int(scene_top_k))
        self.scene_bank: List[Dict[str, Any]] = []
        self.bank_clock: int = 0
        self.active_episode_tag: Optional[str] = None
        self.scene_hit_count: int = 0
        self.scene_miss_count: int = 0
        # ===== 帧级置信度（SmeC-lite，可选）=====
        self.use_frame_confidence = bool(use_frame_confidence)
        self.memory_conf_tau = float(memory_conf_tau)
        self.memory_conf_max = float(memory_conf_max)
        self.memory_writeback_conf_threshold = float(memory_writeback_conf_threshold)
        self.scene_key_lang_weight = float(scene_key_lang_weight)
        self.scene_bank_ddp_sync = bool(scene_bank_ddp_sync)
        self.short_conf: Optional[torch.Tensor] = None  # [N_short]
        self.long_conf: Optional[torch.Tensor] = None  # [N_long]，与 long_memory 同步长度
        self.last_frame_rep: Optional[torch.Tensor] = None  # [D] detach，用于 v_t 代理
        self.scene_episode_text_key: Optional[torch.Tensor] = None  # [D] CPU，用于 write-back key
        # 分阶段评测：帧池未启用时，用当前步 token 均值作为 Scene 写回 fallback
        self._eval_scene_fallback_key: Optional[torch.Tensor] = None
        self._eval_scene_fallback_value: Optional[torch.Tensor] = None

    def configure_inference_memory_pool(self, pool_max_frames: int) -> None:
        """
        仅推理：短期记忆 8 帧、长期记忆 24 帧（见 INFERENCE_MEMORY_SHORT_FRAMES /
        INFERENCE_MEMORY_LONG_FRAMES），max_frames 为二者之和；pool_max_frames > 0
        时生效并清空池内状态。
        """
        if int(pool_max_frames) <= 0:
            return
        self.short_capacity = INFERENCE_MEMORY_SHORT_FRAMES
        self.long_capacity = INFERENCE_MEMORY_LONG_FRAMES
        self.max_frames = self.short_capacity + self.long_capacity
        self.reset()

    def reset(self):
        """Reset frame/token memory; keep scene bank."""
        self.short_memory = None
        self.long_memory = None
        self.long_age = None
        self.long_usage = None
        self.patch_len = None
        self.short_conf = None
        self.long_conf = None
        self.last_frame_rep = None
        self.scene_episode_text_key = None
        self.active_episode_tag = None
        self._eval_scene_fallback_key = None
        self._eval_scene_fallback_value = None

    def clear_scene_bank(self):
        """Clear cross-episode scene bank."""
        self.scene_bank = []
        self.bank_clock = 0
        self.scene_hit_count = 0
        self.scene_miss_count = 0
        self.active_episode_tag = None

    def _frame_reps_from_tokens(
        self, image_embeds: torch.Tensor, tokens_per_image: List[int]
    ) -> Optional[torch.Tensor]:
        if (
            image_embeds is None
            or image_embeds.numel() == 0
            or not tokens_per_image
            or sum(tokens_per_image) != image_embeds.shape[0]
        ):
            return None
        reps: List[torch.Tensor] = []
        start = 0
        for nt in tokens_per_image:
            end = start + nt
            if end > start:
                reps.append(image_embeds[start:end].mean(dim=0))
            start = end
        if not reps:
            return None
        return torch.stack(reps, dim=0)

    def build_scene_key(
        self,
        frame_reps: torch.Tensor,
        text_key: Optional[torch.Tensor] = None,
        lang_weight: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        if frame_reps is None or frame_reps.numel() == 0:
            return None
        if frame_reps.dim() == 1:
            frame_reps = frame_reps.unsqueeze(0)
        vis = frame_reps.float().mean(dim=0)
        lw = self.scene_key_lang_weight if lang_weight is None else float(lang_weight)
        if text_key is not None and text_key.numel() > 0 and lw > 0.0:
            tv = text_key.float().reshape(-1)
            tv = tv / (torch.norm(tv, p=2) + 1e-8)
            vis = vis / (torch.norm(vis, p=2) + 1e-8)
            w = max(0.0, min(1.0, lw))
            key = (1.0 - w) * vis + w * tv.to(device=vis.device, dtype=vis.dtype)
        else:
            key = vis
        key = key / (torch.norm(key, p=2) + 1e-8)
        return key.detach()

    def build_scene_key_from_tokens(
        self,
        image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        text_key: Optional[torch.Tensor] = None,
        lang_weight: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        reps = self._frame_reps_from_tokens(image_embeds, tokens_per_image)
        return self.build_scene_key(reps, text_key=text_key, lang_weight=lang_weight)

    def encode_scene_value(self) -> Optional[torch.Tensor]:
        frame_mem = self.get_frame_memory()
        if frame_mem is not None and frame_mem.numel() > 0:
            return frame_mem.float().mean(dim=0).detach()
        if (
            self._eval_scene_fallback_value is not None
            and self._eval_scene_fallback_value.numel() > 0
        ):
            return self._eval_scene_fallback_value.float().detach()
        return None

    @torch.no_grad()
    def set_eval_scene_fallback_from_tokens(
        self,
        image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        text_key: Optional[torch.Tensor] = None,
        lang_weight: Optional[float] = None,
    ) -> None:
        """分阶段评测早期：帧池未写入时，用当前步视觉 token 更新 Scene 写回 fallback。"""
        key = self.build_scene_key_from_tokens(
            image_embeds,
            tokens_per_image,
            text_key=text_key,
            lang_weight=lang_weight,
        )
        reps = self._frame_reps_from_tokens(image_embeds, tokens_per_image)
        if key is None or reps is None:
            return
        self._eval_scene_fallback_key = key.detach().cpu()
        self._eval_scene_fallback_value = reps.float().mean(dim=0).detach().cpu()

    def decode_scene_value(self, value: torch.Tensor) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.dim() == 1:
            return value
        if value.dim() == 2 and value.shape[0] == 1:
            return value.squeeze(0)
        return value.float().mean(dim=0)

    def retrieve_scene_prior(
        self, query_key: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], float, int]:
        if (
            query_key is None
            or query_key.numel() == 0
            or len(self.scene_bank) == 0
        ):
            self.scene_miss_count += 1
            return None, float("-inf"), -1
        query = query_key.float()
        query = query / (torch.norm(query, p=2) + 1e-8)
        keys = torch.stack([rec["key"].to(query.device) for rec in self.scene_bank], dim=0)
        sims = torch.matmul(keys, query)
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        if best_sim < self.scene_retrieval_threshold:
            self.scene_miss_count += 1
            return None, best_sim, best_idx
        self.bank_clock += 1
        self.scene_bank[best_idx]["last_used"] = self.bank_clock
        self.scene_bank[best_idx]["usage"] += 1
        self.scene_hit_count += 1
        return self.scene_bank[best_idx]["value"], best_sim, best_idx

    def apply_warm_start(
        self,
        image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        scene_value: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if image_embeds is None or image_embeds.numel() == 0 or not tokens_per_image:
            return image_embeds, None
        prior = self.decode_scene_value(scene_value)
        if prior is None or prior.numel() == 0:
            return image_embeds, None
        tau_v = self.scene_warm_tau if tau is None else float(tau)
        tau_v = max(0.0, min(1.0, tau_v))
        first_tokens = int(tokens_per_image[0]) if len(tokens_per_image) > 0 else 0
        if first_tokens <= 0 or first_tokens > image_embeds.shape[0]:
            return image_embeds, None
        x0 = image_embeds[:first_tokens].mean(dim=0)
        h_init = (1.0 - tau_v) * x0 + tau_v * prior.to(device=x0.device, dtype=x0.dtype)
        delta = (h_init - x0).unsqueeze(0)
        warmed = image_embeds.clone()
        warmed[:first_tokens] = warmed[:first_tokens] + delta
        return warmed, h_init

    def write_back_scene_entry(
        self, key: torch.Tensor, value: torch.Tensor, episode_tag: Optional[str] = None
    ) -> bool:
        if key is None or value is None or key.numel() == 0 or value.numel() == 0:
            return False
        key = key.float().detach()
        key = key / (torch.norm(key, p=2) + 1e-8)
        value = value.float().detach()
        self.bank_clock += 1
        entry = {
            "key": key.cpu(),
            "value": value.cpu(),
            "last_used": self.bank_clock,
            "usage": 1,
            "episode_tag": str(episode_tag) if episode_tag is not None else None,
        }
        if len(self.scene_bank) < self.scene_bank_capacity:
            self.scene_bank.append(entry)
            return True
        lru_idx = int(np.argmin([rec["last_used"] for rec in self.scene_bank]))
        self.scene_bank[lru_idx] = entry
        return True

    def finalize_episode_write_back(self, episode_tag: Optional[str] = None) -> bool:
        if self.use_frame_confidence and self.short_conf is not None and self.short_conf.numel() > 0:
            sc = float(self.short_conf.float().mean().item())
            lc = (
                float(self.long_conf.float().mean().item())
                if self.long_conf is not None and self.long_conf.numel() > 0
                else sc
            )
            mean_conf = 0.5 * (sc + lc)
            if mean_conf < self.memory_writeback_conf_threshold:
                if os.environ.get("MEM_DEBUG", "0") == "1":
                    print(
                        f"[MEM_DEBUG] skip scene write-back: mean_conf={mean_conf:.4f} "
                        f"< threshold={self.memory_writeback_conf_threshold}"
                    )
                return False
        scene_value = self.encode_scene_value()
        frame_mem = self.get_frame_memory()
        text_k = self.scene_episode_text_key
        scene_key = None
        if frame_mem is not None:
            tk = None
            if text_k is not None:
                tk = text_k.to(device=frame_mem.device, dtype=frame_mem.dtype)
            scene_key = self.build_scene_key(frame_mem, text_key=tk)
        elif self._eval_scene_fallback_key is not None:
            scene_key = self._eval_scene_fallback_key
        ok = self.write_back_scene_entry(scene_key, scene_value, episode_tag=episode_tag)
        if ok and self.scene_bank_ddp_sync:
            self.sync_scene_bank_ddp_merge()
        if ok:
            self.maybe_log_scene_bank_stats("after_writeback")
        self.scene_episode_text_key = None
        self._eval_scene_fallback_key = None
        self._eval_scene_fallback_value = None
        return ok

    def get_scene_bank_key_diversity(self) -> Optional[float]:
        """返回 bank 内 key 两两余弦相似度均值；样本少则返回 None。"""
        n = len(self.scene_bank)
        if n < 2:
            return None
        keys = torch.stack([rec["key"].float() for rec in self.scene_bank], dim=0)  # [N, D]
        keys = keys / (torch.norm(keys, dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(keys, keys.T)
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return float(sim.masked_select(mask).mean().item())

    def maybe_log_scene_bank_stats(self, tag: str = "") -> None:
        if os.environ.get("MEM_DEBUG", "0") != "1":
            return
        div = self.get_scene_bank_key_diversity()
        extra = f" {tag}" if tag else ""
        print(
            f"[MEM_DEBUG] scene_bank_stats{extra}: size={len(self.scene_bank)}, "
            f"hit={self.scene_hit_count}, miss={self.scene_miss_count}, "
            f"key_pairwise_cos_mean={div}"
        )

    @staticmethod
    def _merge_scene_bank_entries(
        entries: Sequence[Dict[str, Any]],
        capacity: int,
        dedupe_sim: float = 0.995,
    ) -> List[Dict[str, Any]]:
        """按 last_used 从新到旧合并，近似去重后截断到 capacity。"""
        flat = [dict(e) for e in entries]
        flat.sort(key=lambda e: -int(e.get("last_used", 0)))
        kept: List[Dict[str, Any]] = []
        for e in flat:
            if len(kept) >= capacity:
                break
            k = e["key"].float().unsqueeze(0)
            k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-8)
            dup = False
            for o in kept:
                ok = o["key"].float().unsqueeze(0)
                ok = ok / (torch.norm(ok, dim=-1, keepdim=True) + 1e-8)
                if float(torch.sum(k * ok).item()) >= dedupe_sim:
                    dup = True
                    break
            if not dup:
                kept.append(e)
        return kept[:capacity]

    def sync_scene_bank_ddp_merge(self) -> None:
        """DDP：各 rank 的 scene_bank 经 all_gather_object 合并去重（各 rank 结果一致）。"""
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        world = dist.get_world_size()
        if world <= 1:
            return
        gathered: List[Any] = [None] * world
        dist.all_gather_object(gathered, list(self.scene_bank))
        merged_list: List[Dict[str, Any]] = []
        for bank in gathered:
            if bank:
                merged_list.extend(bank)
        self.scene_bank = self._merge_scene_bank_entries(
            merged_list, self.scene_bank_capacity, dedupe_sim=0.995
        )
    
    def get_frame_memory(self):
        """
        Get current frame-level memory [num_frames, D], for retrieval only.
        实际内部存的是 [N, P, D] 的 patch 级帧记忆，这里在帧维度上做一次平均池化。
        """
        mem_list = []
        if self.short_memory is not None and self.short_memory.numel() > 0:
            # [N_short, P, D] -> [N_short, D]
            mem_list.append(self.short_memory.mean(dim=1))
        if self.long_memory is not None and self.long_memory.numel() > 0:
            # [N_long, P, D] -> [N_long, D]
            mem_list.append(self.long_memory.mean(dim=1))
        if not mem_list:
            return None
        return torch.cat(mem_list, dim=0)

    def export_memory_tokens(
        self,
        max_frames: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        将当前帧级记忆池（short + long，内部 [N, P, D]）导出为可以直接喂给 LLM 的视觉 token。

        返回:
            若存在记忆: [F_sel * P, D]
            若无记忆: None

        策略:
            1. 长期记忆按 age 从小到大（越新越优先）排序后放前面。
            2. 短期记忆（时间顺序）紧随其后。
            3. 若总帧数 > max_frames，截断到 max_frames 帧。
        """
        if (self.short_memory is None or self.short_memory.numel() == 0) and (
            self.long_memory is None or self.long_memory.numel() == 0
        ):
            return None

        frames = []

        # 长期记忆：按 age 从小到大排序（越新越优先）
        if self.long_memory is not None and self.long_memory.numel() > 0:
            if self.long_age is not None and self.long_age.shape[0] == self.long_memory.shape[0]:
                _, order = torch.sort(self.long_age, dim=0, descending=False)
                long_sorted = self.long_memory[order]
            else:
                long_sorted = self.long_memory
            frames.append(long_sorted)

        # 短期记忆：按时间顺序保留
        if self.short_memory is not None and self.short_memory.numel() > 0:
            frames.append(self.short_memory)

        all_frames = torch.cat(frames, dim=0)  # [F_total, P, D]
        F_total, P, D = all_frames.shape

        if max_frames is not None and max_frames > 0 and F_total > max_frames:
            all_frames = all_frames[-max_frames:]
            F_total = all_frames.shape[0]

        return all_frames.reshape(F_total * P, D)
    
    @torch.no_grad()
    def update_frame_memory(self, frame_patches: torch.Tensor):
        """
        按帧更新帧级记忆池：
        - 新帧优先进入短期池；
        - 短期池满时，将最老的一帧推入长期池（做 summary），再把新帧放入短期池末尾。
        
        Args:
            frame_patches: 单帧所有 patch 的表示 [P, D]
        """
        if frame_patches is None or frame_patches.numel() == 0:
            return
        # 统一为 [P, D]
        if frame_patches.dim() == 3 and frame_patches.shape[0] == 1:
            frame_patches = frame_patches.squeeze(0)
        if frame_patches.dim() != 2:
            raise ValueError(f"frame_patches must be 2D [P, D], got shape {frame_patches.shape}")
        frame_patches = frame_patches.detach()

        P, D = frame_patches.shape

        new_rep = frame_patches.mean(dim=0).detach()
        v_t: Optional[float] = None
        if self.use_frame_confidence:
            if self.last_frame_rep is not None:
                sim = F.cosine_similarity(
                    new_rep.unsqueeze(0), self.last_frame_rep.unsqueeze(0), dim=-1
                )
                v_t = float((1.0 - sim.clamp(-1.0, 1.0)).item())
                v_t = max(0.0, min(1.0, v_t))
            else:
                v_t = 1.0
            self.last_frame_rep = new_rep.clone()

        # 记录或校验 patch 长度
        if self.patch_len is None:
            self.patch_len = P
        elif P != self.patch_len:
            # 为避免不同 patch 数导致维度不一致，这里保守跳过写入，仅在 MEM_DEBUG 下打印一次。
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] skip_update_frame_memory due to patch_len mismatch: "
                      f"got {P}, expected {self.patch_len}")
            return

        # 初始化短期池
        if self.short_memory is None:
            self.short_memory = frame_patches.unsqueeze(0)  # [1, P, D]
            if self.use_frame_confidence and v_t is not None:
                self.short_conf = torch.tensor(
                    [min(self.memory_conf_max, 1.0 + float(v_t))],
                    device=self.short_memory.device,
                    dtype=torch.float32,
                )
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] init short_memory=1, long_memory=0")
            return

        # 短期池未满：直接追加
        if self.short_memory.shape[0] < self.short_capacity:
            self.short_memory = torch.cat(
                [
                    self.short_memory,
                    frame_patches.unsqueeze(0).to(device=self.short_memory.device, dtype=self.short_memory.dtype),
                ],
                dim=0,
            )
            if self.use_frame_confidence and v_t is not None:
                prev_m = (
                    float(self.short_conf.float().mean().item())
                    if self.short_conf is not None
                    else 1.0
                )
                new_c = min(self.memory_conf_max, prev_m + float(v_t))
                tail = torch.tensor(
                    [new_c], device=self.short_memory.device, dtype=torch.float32
                )
                self.short_conf = (
                    torch.cat([self.short_conf, tail])
                    if self.short_conf is not None
                    else tail
                )
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] short_append size={self.short_memory.shape[0]}, "
                      f"short_cap={self.short_capacity}, long_size={0 if self.long_memory is None else self.long_memory.shape[0]}")
            return

        # 短期池已满：将最老的一帧推入长期池，再滚动短期池并追加当前帧
        oldest = self.short_memory[0:1]  # [1, P, D]
        self._update_long_memory(oldest.squeeze(0))
        # 短期池滚动：丢掉最早一帧，追加当前帧
        self.short_memory = torch.cat(
            [
                self.short_memory[1:],
                frame_patches.unsqueeze(0).to(device=self.short_memory.device, dtype=self.short_memory.dtype),
            ],
            dim=0,
        )
        if self.use_frame_confidence and v_t is not None and self.short_conf is not None:
            prev_m = float(self.short_conf.float().mean().item())
            new_c = min(self.memory_conf_max, prev_m + float(v_t))
            tail = torch.tensor(
                [new_c], device=self.short_memory.device, dtype=torch.float32
            )
            self.short_conf = torch.cat([self.short_conf[1:], tail])
        if os.environ.get("MEM_DEBUG", "0") == "1":
            print(f"[MEM_DEBUG] short_roll size={self.short_memory.shape[0]}, "
                  f"short_cap={self.short_capacity}, long_size={0 if self.long_memory is None else self.long_memory.shape[0]}")

    def _update_long_memory(self, frame_patches: torch.Tensor):
        """
        更新长期记忆池：
        - 未满时直接追加；
        - 满时对 top-k 最相似槽位做加权合并，且 alpha 随 age 衰减。
        """
        if frame_patches is None or frame_patches.numel() == 0:
            return
        # 统一为 [P, D]
        if frame_patches.dim() == 3 and frame_patches.shape[0] == 1:
            frame_patches = frame_patches.squeeze(0)
        if frame_patches.dim() != 2:
            raise ValueError(f"frame_patches must be 2D [P, D], got shape {frame_patches.shape}")
        frame_patches = frame_patches.detach()

        P, D = frame_patches.shape

        # 记录或校验 patch 长度
        if self.patch_len is None:
            self.patch_len = P
        elif P != self.patch_len:
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] skip_update_long_memory due to patch_len mismatch: "
                      f"got {P}, expected {self.patch_len}")
            return

        # 初始化长期池
        if self.long_memory is None:
            self.long_memory = frame_patches.unsqueeze(0)  # [1, P, D]
            self.long_age = torch.zeros(1, dtype=torch.long)
            self.long_usage = torch.ones(1, dtype=torch.long)
            if self.use_frame_confidence:
                self.long_conf = torch.ones(
                    1, dtype=torch.float32, device=self.long_memory.device
                )
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] init long_memory=1 (cap={self.long_capacity})")
            return

        M = self.long_memory.shape[0]
        device = self.long_memory.device
        dtype = self.long_memory.dtype

        # 未满：直接 append
        if M < self.long_capacity:
            self.long_memory = torch.cat(
                [
                    self.long_memory,
                    frame_patches.unsqueeze(0).to(device=device, dtype=dtype),
                ],
                dim=0,
            )
            self.long_age = torch.cat([self.long_age, torch.zeros(1, dtype=torch.long)], dim=0)
            self.long_usage = torch.cat([self.long_usage, torch.ones(1, dtype=torch.long)], dim=0)
            if self.use_frame_confidence:
                if self.long_conf is None or self.long_conf.shape[0] != M:
                    self.long_conf = torch.ones(
                        M, dtype=torch.float32, device=device
                    )
                z = torch.zeros(1, dtype=torch.float32, device=device)
                self.long_conf = torch.cat([self.long_conf, z], dim=0)
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(f"[MEM_DEBUG] long_append size={self.long_memory.shape[0]}, cap={self.long_capacity}")
            return

        # 已满：与长期池中 top-k 最相似槽位加权合并，并使用自适应 alpha
        # 相似度在“帧级别”上计算：先对 patch 做平均池化得到 [M, D]，query 也是 [1, D]
        long_frame_reps = self.long_memory.mean(dim=1)  # [M, D]
        query_rep = frame_patches.mean(dim=0, keepdim=True)  # [1, D]
        sim = self._compute_cosine_similarity(query_rep, long_frame_reps).squeeze(0)  # [M]
        top_k = min(self.merge_top_k, M)
        topk_sim, topk_idx = torch.topk(sim, k=top_k, dim=-1)

        # 未被选中的槽位 age += 1
        all_idx = torch.arange(M, dtype=torch.long)
        mask_selected = torch.zeros_like(all_idx, dtype=torch.bool)
        mask_selected[topk_idx] = True
        self.long_age[~mask_selected] += 1

        # 合并时 alpha 由 age + similarity 共同决定
        # age 大 → alpha_age 小 → 倾向覆盖旧内容
        # sim 高 → alpha_sim 小 → 确认同场景，用新帧替换
        # sim 低 → alpha_sim 大 → 不太相关，保护旧记忆
        alpha_min, alpha_max, decay = 0.05, 0.8, 0.04
        sim_for_merge = {idx.item(): topk_sim[j].item() for j, idx in enumerate(topk_idx)}
        new_vec = frame_patches.to(device=device, dtype=torch.float32)  # [P, D]
        for idx in topk_idx:
            i = idx.item()
            old = self.long_memory[i].float()  # [P, D]
            age_i = int(self.long_age[i].item())
            sim_i = sim_for_merge[i]

            alpha_age = max(alpha_min, alpha_max - decay * age_i)
            alpha_sim = (1.0 - sim_i) * alpha_max
            alpha_i = max(alpha_min, min(alpha_max, 0.4 * alpha_age + 0.6 * alpha_sim))

            # 如果出现 patch 数不一致，保守只在对齐长度内做融合
            if old.shape[0] != new_vec.shape[0]:
                P_min = min(old.shape[0], new_vec.shape[0])
                old_slice = old[:P_min]
                new_slice = new_vec[:P_min]
                updated_slice = alpha_i * old_slice + (1.0 - alpha_i) * new_slice
                old[:P_min] = updated_slice
                updated = old
            else:
                updated = alpha_i * old + (1.0 - alpha_i) * new_vec

            self.long_memory[i] = updated.to(device=device, dtype=dtype)
            # 重置 age，累加 usage
            self.long_age[i] = 0
            self.long_usage[i] += 1
        if os.environ.get("MEM_DEBUG", "0") == "1":
            ages = self.long_age.tolist()
            print(f"[MEM_DEBUG] long_merge size={self.long_memory.shape[0]}, cap={self.long_capacity}, ages={ages}")
    
    def _compute_cosine_similarity(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        计算查询与记忆键之间的余弦相似度。
        
        Args:
            query: [L, D] 或 [1, D]
            keys: [M, D]
        
        Returns:
            similarity: [L, M] 或 [1, M]
        """
        # 保存 device 和 dtype
        device = query.device
        dtype = query.dtype
        
        # 转换为 numpy 并归一化（先转为 float32，因为 numpy 不支持 BFloat16）
        query_np = query.detach().cpu().float().numpy()
        keys_np = keys.detach().cpu().float().numpy()
        
        query_norm = query_np / (norm(query_np, axis=-1, keepdims=True) + 1e-8)
        keys_norm = keys_np / (norm(keys_np, axis=-1, keepdims=True) + 1e-8)
        
        # 余弦相似度 = 点积（归一化后）
        similarity = np.dot(query_norm, keys_norm.T)
        
        # 转换回 torch.Tensor（保持原始 dtype）
        return torch.from_numpy(similarity).to(device=device, dtype=dtype)
    
    def _prepare_batch_weighted(self, samples: List[torch.Tensor], attention_weights: torch.Tensor) -> torch.Tensor:
        """
        根据注意力权重对记忆样本进行加权平均。
        
        Args:
            samples: List of [D] tensors
            attention_weights: [num_samples] 权重（余弦相似度分数）
        
        Returns:
            weighted_ensemble: [D]
        """
        # 转换为 numpy 并使用 temperature 缩放（先转为 float32，因为 numpy 不支持 BFloat16）
        attention_weights_np = attention_weights.detach().cpu().float().numpy()
        attention_weights_np = attention_weights_np / self.temperature
        attention_weights_np = np.exp(attention_weights_np) / np.sum(np.exp(attention_weights_np))
        
        # 加权平均
        device = samples[0].device
        dtype = samples[0].dtype
        samples_np = [s.detach().cpu().float().numpy() for s in samples]
        
        ensemble_prediction = samples_np[0] * attention_weights_np[0]
        for i in range(1, len(samples_np)):
            ensemble_prediction = ensemble_prediction + samples_np[i] * attention_weights_np[i]
        
        return torch.from_numpy(ensemble_prediction).to(device=device, dtype=dtype)

    def fuse_with_memory_per_frame(
        self,
        current_tokens: torch.Tensor,
        tokens_per_image: List[int],
        memory_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        按“帧”为单位进行显式记忆融合。
        memory_override: 若提供 [M, D]，则用其代替短期/长期帧池做只读检索。

        思路：
        - 先对每一帧的 token 做平均池化，得到 [num_frames, D] 的帧级表示。
        - 在帧级别上与显式记忆做一次融合。
        - 将帧级融合后的偏移量广播回该帧内所有 token：
              frame_delta = enhanced_frame - frame_avg
              token = token + frame_delta
        这样既按“帧”检索记忆，又保留帧内空间结构差异。
        """
        if current_tokens is None or current_tokens.numel() == 0:
            return current_tokens

        if not tokens_per_image:
            return current_tokens

        # 检索用记忆：优先 memory_override，否则使用短期+长期帧级记忆。
        mem = memory_override
        if mem is None:
            mem = self.get_frame_memory()
        if mem is None or mem.numel() == 0:
            return current_tokens

        # 校验 tokens_per_image 总和与当前 token 数量一致（不一致则跳过融合）
        total_tokens = sum(tokens_per_image)
        if total_tokens != current_tokens.shape[0]:
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(
                    "[MEM_DEBUG] fuse_with_memory_per_frame: skip fusion, "
                    f"sum(tokens_per_image)={total_tokens} != L={current_tokens.shape[0]}",
                    flush=True,
                )
            return current_tokens

        L, D = current_tokens.shape
        M = mem.shape[0]
        top_k = min(self.top_k, M)

        # 1) 计算每帧的平均特征 [num_frames, D]
        frame_reps = []
        start = 0
        for num_tokens in tokens_per_image:
            end = start + num_tokens
            frame_tokens = current_tokens[start:end]  # [num_tokens, D]
            # 避免空帧
            if frame_tokens.numel() == 0:
                frame_reps.append(torch.zeros(D, device=current_tokens.device, dtype=current_tokens.dtype))
            else:
                frame_reps.append(frame_tokens.mean(dim=0))
            start = end
        frame_reps = torch.stack(frame_reps, dim=0)  # [F, D]

        # 2) 在帧级别上做一次记忆融合
        cosine_sim = self._compute_cosine_similarity(frame_reps, mem)  # [F, M]

        topk_sim, topk_indices = torch.topk(cosine_sim, k=top_k, dim=-1)  # [F, top_k]

        eff_ratio = float(self.update_ratio)
        if self.use_frame_confidence:
            conf_vals: List[torch.Tensor] = []
            if self.short_conf is not None and self.short_conf.numel() > 0:
                conf_vals.append(self.short_conf.float().mean())
            if self.long_conf is not None and self.long_conf.numel() > 0:
                conf_vals.append(self.long_conf.float().mean())
            if conf_vals:
                mean_c = float(torch.stack(conf_vals).mean().item())
                norm = max(self.memory_conf_max, 1e-6)
                tau = max(self.memory_conf_tau, 1e-6)
                gate = min(1.0, mean_c / norm) * min(1.0, mean_c / tau)
                eff_ratio = float(self.update_ratio) * float(gate)

        enhanced_frame_reps = []
        for i in range(frame_reps.shape[0]):
            topk_memory = mem[topk_indices[i]]  # [top_k, D]
            topk_sim_i = topk_sim[i]  # [top_k]

            weighted_memory = self._prepare_batch_weighted(
                [topk_memory[j] for j in range(top_k)],
                topk_sim_i,
            )  # [D]

            enhanced = frame_reps[i] * (1 - eff_ratio) + weighted_memory * eff_ratio
            enhanced_frame_reps.append(enhanced)

        enhanced_frame_reps = torch.stack(enhanced_frame_reps, dim=0)  # [F, D]

        # 3) 将帧级偏移量广播回各帧内部的所有 token（无 in-place，用 torch.cat 拼接）
        out_parts: List[torch.Tensor] = []
        start = 0
        for frame_idx, num_tokens in enumerate(tokens_per_image):
            end = start + num_tokens
            if end > current_tokens.shape[0]:
                end = current_tokens.shape[0]
            if end <= start:
                start = end
                continue

            frame_tokens = current_tokens[start:end]          # [num_tokens, D]
            frame_avg = frame_reps[frame_idx]                 # [D]
            frame_enhanced = enhanced_frame_reps[frame_idx]   # [D]
            delta = (frame_enhanced - frame_avg).unsqueeze(0) # [1, D]

            out_parts.append(frame_tokens + delta)
            start = end

        if not out_parts:
            return current_tokens
        return torch.cat(out_parts, dim=0)

    def process_chunked_with_memory(
        self,
        image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        Chunk-wise memory update and fusion (no frame sampling, no in-place ops).

        Frame sampling should be done upstream (e.g. in data loading via
        two_level_ebbinghaus_indices). This method simply:
        1. Divides frames into chunks.
        2. Writes ALL frames from each chunk into the memory pool.
        3. Uses the accumulated memory to enhance each chunk via read-only fusion.

        Args:
            image_embeds: [L, D], L = sum(tokens_per_image)
            tokens_per_image: token count per frame
            chunk_size: frames per chunk

        Returns:
            new tensor [L, D] built by torch.cat of (possibly enhanced) chunks
        """
        if image_embeds is None or image_embeds.numel() == 0 or not tokens_per_image:
            return image_embeds
        if sum(tokens_per_image) != image_embeds.shape[0]:
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(
                    "[MEM_DEBUG] process_chunked_with_memory: skip chunk path, "
                    f"sum(tokens_per_image)={sum(tokens_per_image)} "
                    f"!= L={image_embeds.shape[0]}",
                    flush=True,
                )
            return image_embeds

        L, D = image_embeds.shape
        F = len(tokens_per_image)
        if chunk_size <= 0:
            chunk_size = 8

        chunk_starts = []
        for i in range(0, F, chunk_size):
            n = min(chunk_size, F - i)
            chunk_starts.append((i, n))

        if len(chunk_starts) < 2:
            # 单 chunk（常见于评估每步帧数 ≤ chunk_size）：仍应用跨步帧记忆
            # 1) 用「写入本步之前」的帧记忆增强当前各帧；2) 再将本步各帧写入记忆池。
            if os.environ.get("MEM_DEBUG", "0") == "1":
                print(
                    f"[MEM_DEBUG] single_chunk_memory F={F}, chunk_size={chunk_size}, "
                    f"chunks={len(chunk_starts)} (cross-step fuse + update)"
                )
            mem_before = self.get_frame_memory()
            if mem_before is not None and mem_before.numel() > 0:
                out = self.fuse_with_memory_per_frame(
                    image_embeds, tokens_per_image, memory_override=mem_before
                )
            else:
                out = image_embeds
            token_start = 0
            for nt in tokens_per_image:
                s, e = token_start, token_start + nt
                if e <= image_embeds.shape[0] and e > s:
                    self.update_frame_memory(image_embeds[s:e])
                token_start += nt
            return out

        if os.environ.get("MEM_DEBUG", "0") == "1":
            print(f"[MEM_DEBUG] process_chunked_memory F={F}, C={len(chunk_starts)}, chunk_size={chunk_size}")

        enhanced_chunks: List[torch.Tensor] = []
        token_start = 0
        for chunk_idx, (frame_start, num_frames) in enumerate(chunk_starts):
            tokens_in_chunk = tokens_per_image[frame_start : frame_start + num_frames]
            chunk_token_len = sum(tokens_in_chunk)
            chunk_tokens = image_embeds[token_start : token_start + chunk_token_len]

            frame_token_ranges = []
            s = 0
            for nt in tokens_in_chunk:
                e = s + nt
                frame_token_ranges.append((s, e))
                s = e

            for local_idx in range(num_frames):
                if local_idx < len(frame_token_ranges):
                    s_local, e_local = frame_token_ranges[local_idx]
                    if e_local > s_local:
                        frame_patches = chunk_tokens[s_local:e_local]
                        self.update_frame_memory(frame_patches)

            mem = self.get_frame_memory()
            if mem is not None and mem.numel() > 0:
                enhanced = self.fuse_with_memory_per_frame(
                    chunk_tokens,
                    tokens_in_chunk,
                    memory_override=mem,
                )
                enhanced_chunks.append(enhanced)
            else:
                enhanced_chunks.append(chunk_tokens)

            token_start += chunk_token_len

        return torch.cat(enhanced_chunks, dim=0)

    def compress_to_memory_tokens(
        self,
        image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        max_frames: int,
    ) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """
        Differentiable frame compression: F frames → max_frames merged frames.

        Preserves gradient flow by operating on ``image_embeds`` directly.
        Groups contiguous frames into ``max_frames`` groups, then performs
        importance-weighted mean-pooling within each group.  Importance is
        derived from the memory bank's content (detached) so it guides *which*
        frames matter, while the actual token values remain differentiable.

        Args:
            image_embeds: [L, D]  (L = sum(tokens_per_image)), **with gradient**
            tokens_per_image: per-frame token count
            max_frames: target number of output frames

        Returns:
            (compressed_embeds, kept_frame_indices):
            - compressed_embeds: [L', D]  where L' ≈ max_frames * P
            - kept_frame_indices: list of kept frame indices (0..F-1), or None if no compression
        """
        infer_cap = getattr(self, "inference_compress_max_frames", None)
        if infer_cap is not None and int(infer_cap) > 0:
            max_frames = int(infer_cap)
        F = len(tokens_per_image)
        if F <= max_frames or max_frames <= 0:
            return image_embeds, None

        D = image_embeds.shape[-1]

        # -- frame boundaries ---------------------------------------------------
        frame_slices: List[tuple] = []
        start = 0
        for nt in tokens_per_image:
            frame_slices.append((start, start + nt))
            start += nt

        # -- frame importance (detached, for weighting only) --------------------
        importance: Optional[torch.Tensor] = None
        mem = self.get_frame_memory()                      # [M, D] or None
        if mem is not None and mem.numel() > 0:
            frame_reps = torch.stack(
                [image_embeds[s:e].detach().mean(dim=0) for s, e in frame_slices]
            )                                              # [F, D]
            sim = self._compute_cosine_similarity(frame_reps, mem)  # [F, M]
            raw_importance = sim.max(dim=-1).values        # [F]
            # add recency bias: newer frames are more important
            recency = torch.linspace(0.0, 1.0, F,
                                     device=image_embeds.device,
                                     dtype=image_embeds.dtype)
            importance = raw_importance + recency

        # -- contiguous temporal grouping: F frames → max_frames groups ---------
        groups: List[List[int]] = []
        for g in range(max_frames):
            g_start = round(g * F / max_frames)
            g_end   = round((g + 1) * F / max_frames)
            groups.append(list(range(g_start, g_end)))

        # -- per-group: keep only the most important frame -----------------------
        # 最后一组强制保留最新帧（当前观测），其余组按 importance 或 recency 选择
        compressed_parts: List[torch.Tensor] = []
        kept_frame_indices: List[int] = []
        for gi, group in enumerate(groups):
            if not group:
                continue

            if len(group) == 1:
                best = group[0]
            elif gi == len(groups) - 1:
                # 最后一组：强制保留最新帧（当前观测）
                best = group[-1]
            elif importance is not None:
                best = group[importance[group].argmax().item()]
            else:
                best = group[-1]

            kept_frame_indices.append(best)
            s, e = frame_slices[best]
            compressed_parts.append(image_embeds[s:e])

        return torch.cat(compressed_parts, dim=0), kept_frame_indices

    def _distribute_budget_ebbinghaus(
        self,
        num_chunks: int,
        chunk_sizes: List[int],
        total_budget: int,
        s: float = 3.0,
    ) -> List[int]:
        """
        Distribute total frame sampling budget across chunks using Ebbinghaus forgetting curve.

        More recent chunks (higher index) get higher weight and more frames.
        Weight formula: w_j = exp(-(C-1-j) / s), where C-1-j is the "age" of chunk j.

        Args:
            num_chunks: total number of chunks
            chunk_sizes: actual frame count per chunk
            total_budget: total frames to sample across all chunks
            s: Ebbinghaus stability parameter; larger = more uniform distribution

        Returns:
            list of per-chunk frame budgets
        """
        if num_chunks <= 0 or total_budget <= 0:
            return [0] * num_chunks

        # chunk age: index 0 is oldest (age = C-1), index C-1 is newest (age = 0)
        ages = np.array([num_chunks - 1 - j for j in range(num_chunks)], dtype=np.float64)
        weights = np.exp(-ages / max(s, 0.01))
        weights = weights / weights.sum()

        raw = weights * total_budget
        allocations = np.floor(raw).astype(int)

        # cap at chunk's actual frame count
        for i in range(num_chunks):
            allocations[i] = min(allocations[i], chunk_sizes[i])

        # distribute remaining budget: first by largest fractional part,
        # then round-robin to handle cases where capped chunks cause leftovers
        current_total = int(allocations.sum())
        remainder = total_budget - current_total
        if remainder > 0:
            fractions = raw - np.floor(raw)
            order = np.argsort(-fractions)
            for idx in order:
                if remainder <= 0:
                    break
                if allocations[idx] < chunk_sizes[idx]:
                    allocations[idx] += 1
                    remainder -= 1
        while remainder > 0:
            distributed_any = False
            for i in range(num_chunks):
                if remainder <= 0:
                    break
                if allocations[i] < chunk_sizes[i]:
                    allocations[i] += 1
                    remainder -= 1
                    distributed_any = True
            if not distributed_any:
                break

        return allocations.tolist()

    def _compute_global_max_frames(self, F: int) -> int:
        """
        根据样本总帧数 F 自适应决定全局抽帧数。
        抽帧数有意 **不** 受记忆池容量 max_frames 的限制，
        这样当抽帧数 > 池容量时，多出来的帧会触发长期记忆的合并逻辑。

        公式：min(F, a + b * sqrt(F))
        - F=16  -> 12
        - F=50  -> 18
        - F=100 -> 24
        - F=200 -> 32
        """
        if F <= 0:
            return 0
        a = 4.0
        b = 2.0
        raw = a + b * (F ** 0.5)
        return max(1, min(F, int(raw)))



# ========== Frame Sampling (MemoryVLN: first+last enforced) ==========

def select_history_indices(
    num_frames: int,
    max_frames: int = 16,
    s: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))

    if rng is None:
        rng = np.random.default_rng()

    t = num_frames - 1
    candidates = np.arange(0, t)
    delta_T = t - candidates
    weights = np.exp(-delta_T / s)
    weights = weights / weights.sum()

    sampled = rng.choice(
        candidates, size=max_frames - 1, replace=False, p=weights
    )
    sampled = np.sort(sampled)
    return list(sampled) + [t]


def _ensure_first_last_frame_indices(
    indices: List[int],
    num_frames: int,
    max_frames: int,
) -> List[int]:
    if num_frames <= 0:
        return []
    if num_frames == 1:
        return [0]
    if max_frames <= 0:
        return [0, num_frames - 1]
    if max_frames == 1:
        return [num_frames - 1]

    merged = sorted({i for i in indices if 0 <= i < num_frames} | {0, num_frames - 1})

    if len(merged) <= max_frames:
        return merged

    first, last = 0, num_frames - 1
    interior = [x for x in merged if x != first and x != last]
    need_inner = max_frames - 2
    if need_inner <= 0:
        return [first, last]

    if len(interior) <= need_inner:
        return sorted({first, last} | set(interior))

    pos = np.linspace(0, len(interior) - 1, need_inner)
    picked_set = set()
    for p in pos:
        j = int(round(float(p)))
        j = max(0, min(len(interior) - 1, j))
        picked_set.add(interior[j])
    picked = sorted(picked_set)

    k = 0
    while len(picked) < need_inner and k < len(interior):
        if interior[k] not in picked:
            picked.append(interior[k])
        k += 1
    picked = sorted(picked)[:need_inner]

    return sorted({first, last} | set(picked))


def two_level_ebbinghaus_indices(
    num_frames: int,
    max_frames: int,
    chunk_size: int = 8,
    s_inner: float = 5.0,
    s_outer: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))

    if rng is None:
        rng = np.random.default_rng()

    chunks = []
    for i in range(0, num_frames, chunk_size):
        n = min(chunk_size, num_frames - i)
        chunks.append((i, n))

    C = len(chunks)
    chunk_sizes_list = [n for _, n in chunks]

    if C < 2:
        raw = select_history_indices(num_frames, max_frames, s=s_inner, rng=rng)
        return _ensure_first_last_frame_indices(raw, num_frames, max_frames)

    ages = np.array([C - 1 - j for j in range(C)], dtype=np.float64)
    weights = np.exp(-ages / max(s_outer, 0.01))
    weights = weights / weights.sum()

    raw = weights * max_frames
    allocations = np.floor(raw).astype(int)

    for i in range(C):
        allocations[i] = min(allocations[i], chunk_sizes_list[i])

    current_total = int(allocations.sum())
    remainder = max_frames - current_total
    if remainder > 0:
        fractions = raw - np.floor(raw)
        order = np.argsort(-fractions)
        for idx in order:
            if remainder <= 0:
                break
            if allocations[idx] < chunk_sizes_list[idx]:
                allocations[idx] += 1
                remainder -= 1
    while remainder > 0:
        distributed_any = False
        for i in range(C):
            if remainder <= 0:
                break
            if allocations[i] < chunk_sizes_list[i]:
                allocations[i] += 1
                remainder -= 1
                distributed_any = True
        if not distributed_any:
            break

    selected_indices = []
    for chunk_idx, (start, n) in enumerate(chunks):
        M_i = int(allocations[chunk_idx])
        if M_i <= 0:
            continue
        if M_i >= n:
            selected_indices.extend(range(start, start + n))
        else:
            local_indices = select_history_indices(n, M_i, s=s_inner, rng=rng)
            selected_indices.extend([start + li for li in local_indices])

    selected_indices.sort()
    return _ensure_first_last_frame_indices(
        selected_indices, num_frames, max_frames
    )
