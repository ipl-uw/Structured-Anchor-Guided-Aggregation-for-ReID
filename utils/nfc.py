"""
Neighbor Feature Centralization (NFC) — training-free re-ID post-processing.

Adapted from Pose2ID (CVPR 2025).  Apply separately to query and gallery feature
sets before distance computation.

Reference
---------
  NFC.py from https://github.com/Pose2ID (CVPR 2025)
"""

import torch
import torch.nn.functional as F


def _pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance matrix between rows of *a* (N×D) and *b* (M×D)."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
    aa = (a * a).sum(dim=1, keepdim=True)
    bb = (b * b).sum(dim=1, keepdim=True)
    return (aa + bb.t() - 2.0 * a @ b.t()).clamp(min=0.0)


def nfc(feat: torch.Tensor, k1: int = 2, k2: int = 2) -> torch.Tensor:
    """Neighbor Feature Centralization.

    Parameters
    ----------
    feat : Tensor [N, D]  (does not need to be pre-normalized)
    k1   : number of nearest neighbors to consider for each sample
    k2   : mutual-check window — neighbor j is kept only if i appears in
           j's top-k2 neighbors

    Returns
    -------
    Tensor [N, D], L2-normalized.
    """
    feat = F.normalize(feat.float(), p=2, dim=1)

    dist = _pairwise_l2(feat, feat)
    # Exclude self by setting diagonal to a large value
    dist.fill_diagonal_(1e9)

    # Top-k1 nearest neighbors for each sample (indices, ascending distance)
    _, rank = dist.topk(k1, dim=1, largest=False, sorted=True)   # [N, k1]

    # Build mutual-neighbor lists: j is a mutual neighbor of i iff
    # i appears in rank[j][:k2]
    rank_np = rank.tolist()
    mutual: list[list[int]] = []
    for i in range(len(rank_np)):
        mlist = [j for j in rank_np[i] if i in rank_np[j][:k2]]
        mutual.append(mlist)

    # Aggregate mutual-neighbor features
    feat_out = feat.clone()
    for i, neighbors in enumerate(mutual):
        if neighbors:
            feat_out[i] = feat[i] + feat[neighbors].sum(dim=0)

    return F.normalize(feat_out, p=2, dim=1)


def apply_nfc_split(feat: torch.Tensor, num_query: int,
                    k1: int = 2, k2: int = 2) -> torch.Tensor:
    """Apply NFC independently to the query and gallery portions of *feat*.

    Parameters
    ----------
    feat      : Tensor [N, D] — query features first, then gallery
    num_query : number of query samples
    k1, k2    : NFC hyper-parameters (see :func:`nfc`)

    Returns
    -------
    Tensor [N, D] with query and gallery portions independently centralized
    and L2-normalized.
    """
    q = nfc(feat[:num_query].clone(), k1=k1, k2=k2)
    g = nfc(feat[num_query:].clone(), k1=k1, k2=k2)
    return torch.cat([q, g], dim=0)
