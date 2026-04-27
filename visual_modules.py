"""
Anchor-guided visual refinement modules.

  DomainAnchorGenerator                  — M per-image domain anchors from token mean-pool
  CameraConditionedDomainAnchorGenerator — M per-image domain anchors from token + camera ID
  CrossAttentionBlock                    — single cross-attention + FFN block
  SemanticRefinementModule               — stack of CrossAttentionBlock layers
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DomainAnchorGenerator(nn.Module):
    """
    Generates M dynamic domain anchors per image from its own patch token pool.

    The domain embedding is computed by mean-pooling the patch tokens, then a
    small MLP maps this to M anchor vectors in the same space as the tokens.

    This allows the model to adapt to unseen cameras at test time without any
    explicit camera or platform identifier.
    """

    def __init__(self, dim: int, num_anchors: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.dim         = dim
        self.anchor_gen  = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_anchors * dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C]
        Returns domain_anchors [B, M, C]
        """
        domain_emb = tokens.mean(dim=1)                      # [B, C]
        anchors    = self.anchor_gen(domain_emb)             # [B, M*C]
        return anchors.view(-1, self.num_anchors, self.dim)  # [B, M, C]


class CameraConditionedDomainAnchorGenerator(nn.Module):
    """Domain anchors conditioned on both camera ID and image content.

    Combines a per-camera embedding (dataset-level viewpoint/lighting prior)
    with a visual mean-pool of patch tokens (instance-level content) by
    concatenation before the anchor MLP.

    Forward: (tokens [B,N,dim], cam_ids [B]) -> domain_anchors [B, M, dim]
    """

    def __init__(self, dim: int, num_anchors: int, camera_num: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.dim = dim
        self.cam_embed = nn.Embedding(camera_num, dim)
        nn.init.normal_(self.cam_embed.weight, std=0.02)
        self.anchor_gen = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, num_anchors * dim),
        )

    def forward(self, tokens: torch.Tensor, cam_ids: torch.Tensor) -> torch.Tensor:
        """tokens: [B, N, dim]  cam_ids: [B] int64  →  [B, M, dim]"""
        visual   = tokens.mean(dim=1)                       # [B, dim]
        cam      = self.cam_embed(cam_ids)                  # [B, dim]
        combined = torch.cat([visual, cam], dim=1)          # [B, dim*2]
        return self.anchor_gen(combined).view(tokens.shape[0], self.num_anchors, self.dim)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block: query attends to key_value tokens.

      query:     [B, Nq,  C]
      key_value: [B, Nkv, C]
      output:    [B, Nq,  C]

    Attention weights [B, Nq, Nkv] are always returned and used by
    SemanticRefinementModule for attention-weighted pooling.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                             batch_first=True)
        mlp_dim      = int(dim * mlp_ratio)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff      = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        q  = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, weights = self.attn(q, kv, kv,
                                      average_attn_weights=True,
                                      need_weights=True)
        query = query + attn_out
        query = query + self.ff(self.norm_ff(query))
        return query, weights   # weights: [B, Nq, Nkv]


class SemanticRefinementModule(nn.Module):
    """
    Refines image patch tokens by attending to a set of semantic anchor vectors.

    Patch tokens (query) attend to anchors (key/value) through one or more
    CrossAttentionBlock layers.  The last-layer attention weights are returned
    for attention-weighted pooling of the refined tokens.

      tokens:  [B, N, C]  — patch tokens from the frozen backbone
      anchors: [B, K, C]  — concatenated text + domain anchors
      Returns: refined tokens [B, N, C], attention weights [B, N, K]
    """

    def __init__(self, dim: int, num_heads: int,
                 num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tokens: torch.Tensor, anchors: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        last_w = None
        for layer in self.layers:
            tokens, last_w = layer(tokens, anchors)
        return tokens, last_w   # last_w: [B, N, K]


class CLIPGroundedDomainAnchorGenerator(nn.Module):
    """
    Text-grounded variant of DomainAnchorGenerator.

    M learned query vectors cross-attend over the K text anchors conditioned
    on the image's CLIP projection.  Grounds domain anchors in the same
    semantic space as the text anchors while still producing per-image
    (dynamic) context vectors.

    Call site must pass (clip_proj, text_anchors) rather than (tokens).
    """

    def __init__(self, dim: int, num_anchors: int,
                 clip_dim: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.clip_proj   = nn.Linear(clip_dim, dim, bias=False)
        self.queries     = nn.Parameter(torch.zeros(num_anchors, dim))
        nn.init.normal_(self.queries, std=0.02)
        self.norm_q      = nn.LayerNorm(dim)
        self.norm_kv     = nn.LayerNorm(dim)
        self.attn        = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_out    = nn.LayerNorm(dim)

    def forward(self, clip_proj: torch.Tensor,
                text_anchors: torch.Tensor) -> torch.Tensor:
        """
        clip_proj:    [B, clip_dim]  — CLIP image projection (512-d)
        text_anchors: [B, K, dim]   — text anchor vectors from TextAnchorModule
        Returns:      [B, M, dim]   — text-grounded per-image domain anchors
        """
        B       = clip_proj.shape[0]
        cond    = self.clip_proj(clip_proj).unsqueeze(1)        # [B, 1, dim]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, dim]
        queries = queries + cond
        attn_out, _ = self.attn(
            self.norm_q(queries),
            self.norm_kv(text_anchors),
            text_anchors,
        )
        return self.norm_out(queries + attn_out)                # [B, M, dim]
