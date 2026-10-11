import torch
import torch.nn as nn


class PositionEmbeddingSine6D(nn.Module):
    """
    Sinusoidal encoding for 6D pose (e.g. xyz + euler), analogous to PositionEmbeddingSine3D /
    DETR-style image position encodings. Each scalar channel gets ``embedding_size // 6`` frequency bands
    (sin/cos interleaved), then channels are concatenated to length <= ``embedding_size`` (remainder zero-padded).
    """

    def __init__(self, embedding_size: int, temperature: float = 10000.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 6) pose coordinates (already scaled by caller if desired).
        Returns:
            (B, N, embedding_size) float tensor (same dtype/device as x where possible).
        """
        B, N, six = x.shape
        if six != 6:
            raise ValueError(f"PositionEmbeddingSine6D expects last dim 6, got {six}")

        n_axes = 6
        num_feats = max(1, self.embedding_size // n_axes)
        # Effective encoded width per axis matches 3D implementation
        dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)

        axis_embeds = []
        for i in range(n_axes):
            pos_i = x[:, :, i][..., None] / dim_t
            if num_feats % 2 != 0:
                pos_i = torch.cat(
                    [pos_i, torch.zeros(B, N, 1, dtype=pos_i.dtype, device=x.device)], dim=-1
                )
                pos_i = torch.stack(
                    (pos_i[:, :, 0::2].sin(), pos_i[:, :, 1::2].cos()), dim=3
                ).flatten(2)[..., :-1]
            else:
                pos_i = torch.stack(
                    (pos_i[:, :, 0::2].sin(), pos_i[:, :, 1::2].cos()), dim=3
                ).flatten(2)
            axis_embeds.append(pos_i)

        pos = torch.cat(axis_embeds, dim=2)
        out = torch.zeros((B, N, self.embedding_size), dtype=x.dtype, device=x.device)
        c = min(pos.shape[2], self.embedding_size)
        out[:, :, :c] = pos[:, :, :c]
        return out
