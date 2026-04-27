"""`nAnchor-Guided ReID training entrypoint.`n"""

import os
import sys
import warnings
import datetime
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import math
import time
import argparse
from datetime import timedelta
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

# ---------------------------------------------------------------------------
# Resolve repo root so we can import existing dataset / model / loss modules
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from datasets.market1501 import Market1501
from datasets.mmmp import MMMP
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.occ_market import OccludedMarket1501
from datasets.occ_reid import OccludedREID
from datasets.bases import ImageDataset
from datasets.sampler import RandomIdentitySampler
from torch.utils.data import Dataset
from model.make_model_uniprompt import build_transformer, load_clip_to_cpu
from model.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from utils.metrics import R1_mAP_eval
from utils.nfc import apply_nfc_split
from visual_modules import (SemanticRefinementModule, DomainAnchorGenerator,
                            CLIPGroundedDomainAnchorGenerator,
                            CameraConditionedDomainAnchorGenerator)

# ===========================================================================
# Config
# ===========================================================================
CFG: Dict = {
    # --- Dataset ---
    "dataset":        "market1501",
    "dataset_root":   "",
    "checkpoint_dir": "output/structured_anchor_guided",

    # --- Backbone ---
    "img_size":       [256, 128],
    "stride_size":    [12, 12],
    "pixel_mean":     [0.5, 0.5, 0.5],
    "pixel_std":      [0.5, 0.5, 0.5],

    # --- Stage 1: text prompt pre-training ---
    "stage1_epochs":  100,
    "stage1_lr":      3.5e-4,
    "stage1_warmup":  5,       # linear warmup epochs
    "stage1_batch":   64,      # batch size for cached image feature iteration
    "stage1_save_period": 10,  # save checkpoint every N epochs

    # --- Stage 2: backbone fine-tuning ---
    "stage2_epochs":  80,
    "stage2_lr":      5e-6,
    "stage2_milestones": [40, 70],  # WarmupMultiStep decay epochs
    "stage2_warmup_iters": 10,      # iteration-level warmup

    # Stage 2 loss weights (CLIP-ReID defaults)
    "lambda_id":       0.25,
    "lambda_triplet":  1.0,
    "lambda_i2t":      1.0,
    "margin":          0.3,
    "lambda_anc_cons": 0.0,   # per-anchor triplet consistency loss weight
    "i2t_loss":       "xent",   # "xent" or "supcon" (bidirectional SupConLoss, UniPrompt-style)

    # --- Stage 3: visual module fine-tuning on frozen Stage-2 backbone ---
    # Stage 3a: anchor warm-up (refine frozen, only anchors+bn+classifier train)
    "s3a_epochs":         20,
    "s3a_lr":             3.5e-4,
    "s3a_warmup":         5,
    # Stage 3b: full fine-tune (refine unfrozen, lower LR)
    "s3b_epochs":         80,
    "s3b_lr":             2e-4,
    "s3b_warmup":         5,
    "stage3_save_period": 10,
    "s3_num_views":       4,      # augmented views per image cached for Stage 3
    "s3_no_cache":        False,  # skip token cache; extract tokens on-the-fly (slower, less memory)
    "s3_num_anchors":     24,    # learnable semantic anchor vectors
    "use_free_anchors":   False, # replace text-grounded anchors with unconstrained learnable vectors (ablation)
    "s3_num_domain_anchors": 3,  # dynamic per-image domain anchor vectors
    "s3_refine_layers":   2,     # SemanticRefinementModule depth
    "s3_embed_dim":       768,   # output embedding dimension
    "s3_dropout":         0.1,   # dropout in visual modules during Stage 3 (disabled in Stage 4)
    "s3_token_layer":     9,     # which ViT block to take patch tokens from (1-indexed; None = second-to-last)
    "s3_token_layers":    None,  # list of ViT blocks, e.g. [6, 9]; overrides s3_token_layer when set
    "s3_anchor_part_ctx": 12,    # number of frozen phrase tokens per anchor
    "s3_num_classifiers": 4,     # multi-classifier heads

    # --- Stage 4: end-to-end fine-tune (backbone + visual module) ---
    "s4_epochs":          100,
    "s4_backbone_lr":     5e-6,  # same scale as Stage 2
    "s4_head_lr":         2e-5,  # visual module: 10x backbone
    "s4_warmup":          10,
    "stage4_save_period": 10,
    "lambda_aux_backbone": 0.1,  # scale factor for backbone auxiliary loss block in Stage 4
    "enable_stage4":      False, # Stage 4 is opt-in; set true in config to run
    "eval_rerank":        False, # also report re-ranked mAP at final epoch of each stage
    "eval_tta":           False, # also report TTA (horizontal flip) mAP at final epoch of Stage 4

    # --- Feature fusion weights ---
    "fusion_w_refined": 2.0,
    "fusion_w_imgfeat": 0.2,
    "fusion_w_proj":    0.05,
    "fusion_beta":      0.3,

    # --- Domain anchor variant ---
    # False = visual mean-pool MLP (default)
    # True  = CLIP-grounded: M learned queries cross-attend over text anchors,
    #         conditioned on the image's 512-d CLIP projection
    "use_clip_domain_anchors": False,

    # --- Neighbor Feature Centralization (NFC) ---
    "use_nfc": False,   # apply NFC post-processing at eval time
    "nfc_k1":  2,
    "nfc_k2":  2,

    # --- Shared ---
    "batch_size":     64,
    "num_instance":   4,       # images per identity (RandomIdentitySampler)
    "weight_decay":   1e-4,
    "num_workers":    8,
    "eval_period":    10,      # evaluate every N epochs in Stage 2
    "lambda_div":     0.05,
    "lambda_cls_div": 0.5,
    "use_sie_camera": False,
    "sie_coe":        3.0,
    "use_v1_features": False,
}


V1_FEATURE_OVERRIDES: Dict = {
    "stride_size":        [16, 16],
    "stage1_epochs":      30,
    "stage2_epochs":      60,
    "stage2_milestones":  [30, 50],
    "s3a_lr":             5e-4,
    "s3b_epochs":         60,
    "s3b_lr":             3.5e-4,
    "s3_num_anchors":     12,
    "s3_refine_layers":   2,
    "s3_token_layer":     None,
    "s3_anchor_part_ctx": 0,
    "s3_num_classifiers": 1,
    "lambda_div":         0.0,
    "lambda_cls_div":     0.0,
}

# ===========================================================================
# Minimal cfg-like namespace so build_transformer / make_loss can consume it
# ===========================================================================
class _Cfg:
    """Thin wrapper that lets attribute access fall through to a dict."""
    class _NS:
        pass

    def __init__(self, d: Dict):
        # MODEL
        self.MODEL = self._NS()
        self.MODEL.NAME = "ViT-B-16"
        self.MODEL.COS_LAYER = False
        self.MODEL.NECK = "bnneck"
        self.MODEL.METRIC_LOSS_TYPE = "triplet"
        self.MODEL.IF_LABELSMOOTH = "on"
        self.MODEL.NO_MARGIN = False
        self.MODEL.ID_LOSS_WEIGHT = d["lambda_id"]
        self.MODEL.TRIPLET_LOSS_WEIGHT = d["lambda_triplet"]
        self.MODEL.I2T_LOSS_WEIGHT = d["lambda_i2t"]
        self.MODEL.STRIDE_SIZE = d["stride_size"]
        self.MODEL.SIE_CAMERA = d.get("use_sie_camera", False)
        self.MODEL.SIE_VIEW = False
        self.MODEL.SIE_COE = d.get("sie_coe", 3.0)
        self.MODEL.MOE = self._NS()
        self.MODEL.MOE.ENABLED = False
        self.MODEL.DIST_TRAIN = False

        # INPUT
        self.INPUT = self._NS()
        self.INPUT.SIZE_TRAIN = d["img_size"]
        self.INPUT.SIZE_TEST = d["img_size"]
        self.INPUT.PROB = 0.5
        self.INPUT.RE_PROB = 0.5
        self.INPUT.PADDING = 10
        self.INPUT.PIXEL_MEAN = d["pixel_mean"]
        self.INPUT.PIXEL_STD = d["pixel_std"]

        # SOLVER
        self.SOLVER = self._NS()
        self.SOLVER.MARGIN = d["margin"]

        # DATALOADER
        self.DATALOADER = self._NS()
        self.DATALOADER.SAMPLER = "softmax_triplet"
        self.DATALOADER.NUM_INSTANCE = d["num_instance"]
        self.DATALOADER.NUM_WORKERS = d["num_workers"]

        # DATASETS
        self.DATASETS = self._NS()
        self.DATASETS.NAMES = d.get("dataset", "market1501")
        self.DATASETS.ROOT_DIR = d["dataset_root"]
        self.DATASETS.EXP_SETTING = d.get("exp_setting", "")

        # TEST
        self.TEST = self._NS()
        self.TEST.NECK_FEAT = "before"
        self.TEST.FEAT_NORM = "yes"
        self.TEST.IMS_PER_BATCH = d.get("eval_batch_size", 200)


# ===========================================================================
# Data helpers
# ===========================================================================
def _make_transforms(cfg, train: bool):
    if train:
        return T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1),
        ])
    else:
        return T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])


def _train_collate(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    return (torch.stack(imgs), torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64), torch.tensor(viewids, dtype=torch.int64))


def _val_collate(batch):
    imgs, pids, camids, viewids, paths = zip(*batch)
    return (torch.stack(imgs), pids, camids,
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64), paths)


def make_loaders(raw_cfg: Dict, cfg):
    ds_name = raw_cfg.get("dataset", "market1501")
    if ds_name == "mmmp":
        dataset = MMMP(root=raw_cfg["dataset_root"],
                       exp_setting=raw_cfg["exp_setting"],
                       split_root=raw_cfg.get("split_root", None))
    elif ds_name == "dukemtmc":
        dataset = DukeMTMCreID(root=raw_cfg["dataset_root"])
    elif ds_name == "occ_duke":
        dataset = OCC_DukeMTMCreID(root=raw_cfg["dataset_root"])
    elif ds_name == "occ_market":
        dataset = OccludedMarket1501(root=raw_cfg["dataset_root"])
    elif ds_name == "occ_reid":
        dataset = OccludedREID(root=raw_cfg["dataset_root"])
    else:
        dataset = Market1501(root=raw_cfg["dataset_root"])
    train_tfm  = _make_transforms(cfg, train=True)
    val_tfm    = _make_transforms(cfg, train=False)

    nw = raw_cfg["num_workers"]
    pw = nw > 0  # persistent_workers

    if dataset.train:
        # Stage-1 loader: no augmentation, plain shuffle
        stage1_set = ImageDataset(dataset.train, val_tfm)
        stage1_loader = DataLoader(
            stage1_set, batch_size=raw_cfg["batch_size"], shuffle=True,
            num_workers=nw, collate_fn=_train_collate, pin_memory=True, persistent_workers=pw,
        )

        # Stage-2 loader: augmentation + identity sampling
        stage2_set = ImageDataset(dataset.train, train_tfm)
        stage2_loader = DataLoader(
            stage2_set, batch_size=raw_cfg["batch_size"],
            sampler=RandomIdentitySampler(dataset.train, raw_cfg["batch_size"], raw_cfg["num_instance"]),
            num_workers=nw, collate_fn=_train_collate, pin_memory=True, persistent_workers=pw,
        )
    else:
        stage1_loader = stage2_loader = None

    # Val loader: query + gallery
    # num_workers=0 avoids Windows spawn deadlocks, especially during eval-only
    # where training loaders are never used but their 2*nw workers are alive.
    val_set = ImageDataset(dataset.query + dataset.gallery, val_tfm)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=0, collate_fn=_val_collate, pin_memory=True,
    )

    num_train_pids = raw_cfg.get("num_train_classes", dataset.num_train_pids)
    cam_num = dataset.num_train_cams if dataset.train else 0
    return stage1_loader, stage2_loader, val_loader, len(dataset.query), num_train_pids, cam_num


# ===========================================================================
# LR helpers
# ===========================================================================
def _warmup_multistep_lr(optimizer, warmup_iters: int, milestones, gamma: float = 0.1,
                          warmup_factor: float = 0.1):
    """Returns a LambdaLR that replicates WarmupMultiStepLR (iteration-level)."""
    mile_set = set(milestones)
    def _fn(iteration):
        if iteration < warmup_iters:
            alpha = iteration / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        factor = 1.0
        for m in sorted(milestones):
            if iteration >= m:
                factor *= gamma
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_fn)


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def validate(model, val_loader, num_query, device, raw_cfg=None) -> Tuple[float, float]:
    model.eval()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes")
    evaluator.reset()
    use_sie = raw_cfg is not None and raw_cfg.get("use_sie_camera", False)
    for imgs, pids, camids, camids_batch, viewids, _ in val_loader:
        imgs = imgs.to(device)
        cam_label = camids_batch.to(device) if use_sie else None
        feat = model(x=imgs, cam_label=cam_label, view_label=None)
        evaluator.update((feat.cpu(), pids, camids))
    cmc, mAP, *_ = evaluator.compute()
    return float(cmc[0]) * 100, float(mAP) * 100


def _cross_decorr_loss(feat: torch.Tensor, img_feat: torch.Tensor, lambda_orth: float) -> torch.Tensor:
    """Cross-covariance decorrelation between refined feat and backbone CLS.

    Computes the normalized DAD cross-correlation matrix C = FaG / (B-1)
    and penalises CA2.mean() a all DA2 pairs of (refined_dim_i, cls_dim_j).
    Mean-centering + per-dim std normalisation makes this sensitive to
    correlations that cancel in the raw dot-product (which is why the
    plain cosine-similarity loss reads ~0).
    """
    B, D = feat.shape
    f = feat  - feat.mean(dim=0)
    g = img_feat.detach() - img_feat.detach().mean(dim=0)
    f = f / f.std(dim=0).clamp(min=1e-4)
    g = g / g.std(dim=0).clamp(min=1e-4)
    C = (f.T @ g) / max(B - 1, 1)   # [D, D] cross-correlation
    return lambda_orth * C.pow(2).mean()


# ===========================================================================
# Stage 1 a text prompt pre-training
# ===========================================================================
def run_stage1(model, stage1_loader, val_loader, num_query, raw_cfg: Dict, device, ckpt_dir: str):
    epochs       = raw_cfg["stage1_epochs"]
    base_lr      = raw_cfg["stage1_lr"]
    warmup_ep    = raw_cfg["stage1_warmup"]
    s1_batch     = raw_cfg["stage1_batch"]
    save_period  = raw_cfg.get("stage1_save_period", 10)

    print(f"\n[Stage 1] pre-training text prompts for {epochs} epochs  lr={base_lr:.1e}")

    # Only ctx_generic is trainable
    model.enable_stage1a_training()
    for p in model.parameters():
        if p is not model.prompt_learner.ctx_generic:
            p.requires_grad_(False)
    model.prompt_learner.ctx_generic.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [model.prompt_learner.ctx_generic], lr=base_lr, weight_decay=raw_cfg["weight_decay"]
    )

    def _lr_lambda(ep):
        if ep < warmup_ep:
            return 1e-5 / base_lr + (1.0 - 1e-5 / base_lr) * ep / max(warmup_ep, 1)
        # cosine decay to 1e-6
        t = (ep - warmup_ep) / max(epochs - warmup_ep, 1)
        return (1e-6 / base_lr) + (1.0 - 1e-6 / base_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    xent = SupConLoss(device)

    # Pre-compute all image features once (backbone is frozen)
    print("  Pre-extracting image features ...")
    all_img_feats, all_labels, all_views = [], [], []
    model.eval()
    use_sie = raw_cfg.get("use_sie_camera", False)
    with torch.no_grad():
        for imgs, pids, camids, viewids in stage1_loader:
            imgs = imgs.to(device)
            cam_label = camids.to(device) if use_sie else None
            with torch.amp.autocast("cuda"):
                feats = model(x=imgs, get_image=True, cam_label=cam_label)   # [B, 512]
            all_img_feats.append(feats.cpu())
            all_labels.append(pids)
            all_views.append(viewids)

    img_feats = torch.cat(all_img_feats).to(device)   # [N, 512]
    labels    = torch.cat(all_labels).to(device)
    views     = torch.cat(all_views).to(device)
    N = img_feats.shape[0]
    print(f"  Cached {N} image features.")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        perm = torch.randperm(N, device=device)
        for i in range(0, N, s1_batch):
            idx = perm[i : i + s1_batch]
            if idx.shape[0] < 2:
                continue
            img_batch  = img_feats[idx]
            lbl_batch  = labels[idx]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                text_feats = model(label=lbl_batch, get_text=True, view=None)

            loss = xent(img_batch, text_feats, lbl_batch, lbl_batch) + \
                   xent(text_feats, img_batch, lbl_batch, lbl_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step(epoch)
        avg = total_loss / max(n_batches, 1)
        print(f"  [S1 {epoch:3d}/{epochs}]  loss: {avg:.4f}  lr: {optimizer.param_groups[0]['lr']:.2e}", end="")

        if epoch % save_period == 0 or epoch == epochs:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"model_stage1_ep{epoch:03d}.pth"))
            print(f"  -> saved", end="")
        print()

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_stage1.pth"))
    print("[Stage 1] done.")


# ===========================================================================
# Stage 2 a backbone fine-tuning
# ===========================================================================
def _make_stage2_loss(num_classes: int, raw_cfg: Dict, device):
    xent    = CrossEntropyLabelSmooth(num_classes)
    triplet = TripletLoss(raw_cfg["margin"])
    w_id    = raw_cfg["lambda_id"]
    w_tri   = raw_cfg["lambda_triplet"]
    w_i2t   = raw_cfg["lambda_i2t"]

    def loss_fn(score, feat, target, img_feat_proj, text_feats_all):
        id_loss  = xent(score, target) if not isinstance(score, list) else \
                   sum(xent(s, target) for s in score)
        tri_loss = triplet(feat, target)[0] if not isinstance(feat, list) else \
                   sum(triplet(f, target)[0] for f in feat)
        i2t_loss = xent(img_feat_proj @ text_feats_all.t(), target)
        return w_id * id_loss + w_tri * tri_loss + w_i2t * i2t_loss

    return loss_fn


def run_stage2(model, stage2_loader, val_loader, num_query, num_classes: int,
               raw_cfg: Dict, device, ckpt_dir: str):
    epochs     = raw_cfg["stage2_epochs"]
    base_lr    = raw_cfg["stage2_lr"]
    eval_every = raw_cfg["eval_period"]

    print(f"\n[Stage 2] backbone fine-tuning for {epochs} epochs  lr={base_lr:.1e}")

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": base_lr}],
        weight_decay=raw_cfg["weight_decay"]
    )

    # Iteration-level warmup + multi-step decay (mirrors CLIP-ReID solver)
    n_iters_per_epoch = len(stage2_loader)
    milestone_iters = [m * n_iters_per_epoch for m in raw_cfg["stage2_milestones"]]
    scheduler = _warmup_multistep_lr(
        optimizer,
        warmup_iters=raw_cfg["stage2_warmup_iters"],
        milestones=milestone_iters,
    )

    loss_fn = _make_stage2_loss(num_classes, raw_cfg, device)
    scaler  = torch.amp.GradScaler("cuda")

    best_mAP = 0.0
    global_iter = 0

    for epoch in range(1, epochs + 1):
        # Pre-compute all class text features (frozen after Stage 1)
        model.eval()
        s2_batch = raw_cfg["stage1_batch"]
        text_feats_all = []
        with torch.no_grad():
            for i in range(0, num_classes, s2_batch):
                l = torch.arange(i, min(i + s2_batch, num_classes), device=device)
                with torch.amp.autocast("cuda"):
                    tf = model(label=l, get_text=True)
                text_feats_all.append(tf.cpu())
        text_feats_all = torch.cat(text_feats_all).float().to(device)   # [num_classes, 512]

        model.train()
        total_loss, t0 = 0.0, time.time()
        use_sie = raw_cfg.get("use_sie_camera", False)
        for imgs, pids, camids, viewids in stage2_loader:
            imgs   = imgs.to(device)
            target = pids.to(device)
            cam_label = camids.to(device) if use_sie else None

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(x=imgs, label=target, cam_label=cam_label, view_label=None)
                if len(out) == 4:
                    scores, feats_all, img_feat_proj, _ = out
                else:
                    scores, feats_all, img_feat_proj = out

                score = scores[0]
                feat  = feats_all[1]
                loss = loss_fn(score, feat, target, img_feat_proj, text_feats_all)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_iter += 1
            total_loss  += loss.item()

        elapsed = time.time() - t0
        avg = total_loss / max(len(stage2_loader), 1)
        print(f"  [S2 {epoch:3d}/{epochs}]  loss: {avg:.4f}  ({elapsed:.0f}s)", end="")

        if epoch % eval_every == 0 or epoch == epochs:
            r1, mAP = validate(model, val_loader, num_query, device, raw_cfg)
            print(f"  mAP: {mAP:.1f}%  Rank-1: {r1:.1f}%", end="")
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_stage2_best.pth"))
                print("  <- best", end="")
        print()

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_stage2_final.pth"))
    print(f"[Stage 2] done.  Best mAP: {best_mAP:.1f}%")
    return best_mAP


# ===========================================================================
# Stage 3 a visual module fine-tuning on frozen Stage-2 backbone
# ===========================================================================

class _TokenCache(Dataset):
    """Stores pre-extracted backbone patch tokens, CLIP CLS projections, and x12 CLS on CPU.

    tokens shape:
      [N, T, D]    a single view (legacy)
      [N, V, T, D] a multi-view; __getitem__ randomly samples one view
    projs / cls_feats shape (optional):
      [N, D]    a single view
      [N, V, D] a multi-view; same view index as tokens
    """
    def __init__(self, tokens: torch.Tensor, pids: torch.Tensor,
                 projs: torch.Tensor = None, cls_feats: torch.Tensor = None,
                 cam_ids: torch.Tensor = None):
        self.tokens     = tokens      # [N, T, D] or [N, V, T, D]  fp16 CPU
        self.projs      = projs       # [N, D] or [N, V, D]  fp16 CPU; None if not cached
        self.cls_feats  = cls_feats   # [N, D] or [N, V, D]  fp16 CPU; x12 CLS for S2 loss
        self.pids       = pids        # [N]  int64 CPU
        self.cam_ids    = cam_ids     # [N]  int64 CPU; None if not available
        self.multi_view = tokens.dim() == 4

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        cam = self.cam_ids[idx] if self.cam_ids is not None else torch.tensor(-1, dtype=torch.int64)
        if self.multi_view:
            v    = torch.randint(self.tokens.shape[1], (1,)).item()
            tok  = self.tokens[idx, v]
            proj = self.projs[idx, v]     if self.projs     is not None else torch.empty(0)
            cls  = self.cls_feats[idx, v] if self.cls_feats is not None else torch.empty(0)
            return tok, proj, cls, self.pids[idx], cam
        proj = self.projs[idx]     if self.projs     is not None else torch.empty(0)
        cls  = self.cls_feats[idx] if self.cls_feats is not None else torch.empty(0)
        return self.tokens[idx], proj, cls, self.pids[idx], cam


_ANCHOR_PHRASES = [
    "the face and hair of a person",
    "the head and neck of a person",
    "the upper torso of a person",
    "the arms and sleeves of a person",
    "the waist and hips of a person",
    "the legs of a person",
    "the feet and shoes of a person",
    "the upper-body clothing of a person",
    "the lower-body clothing of a person",
    "the outerwear of a person",
    "the bag or backpack carried by a person",
    "the object held in the hand of a person",
    "the hat or headwear of a person",
    "the overall clothing color of a person",
    "the color of the upper-body clothing",
    "the color of the lower-body clothing",
    "the texture and pattern of the clothing",
    "the logos or fine details on the clothing of a person",
    "the overall appearance of a person",
    "the silhouette and body shape of a person",
    "the pose and posture of a person",
    "the glasses or sunglasses worn by a person",
    "the color and style of the shoes of a person",
    "the skin tone and complexion of a person",
]


def _init_ctx_from_phrases(K: int, n_ctx: int, ctx_dim: int,
                            ctx_mean: torch.Tensor,
                            clip_model=None) -> torch.Tensor:
    """Initialise [K, n_ctx, ctx_dim] from CLIP token embeddings of _ANCHOR_PHRASES.

    If n_ctx == 0 or clip_model is None, returns zeros (caller skips the buffer).
    Phrases cycle if K > len(_ANCHOR_PHRASES).
    If a phrase has fewer tokens than n_ctx, remaining positions are filled with
    the phrase mean embedding.
    """
    out = torch.zeros(K, n_ctx, ctx_dim)
    if n_ctx == 0 or clip_model is None:
        return out
    tokenizer = _Tokenizer()
    tok_emb = clip_model.token_embedding.weight.detach().float()  # [vocab, 512]
    for k in range(K):
        phrase = _ANCHOR_PHRASES[k % len(_ANCHOR_PHRASES)]
        toks = tokenizer.encode(phrase)  # list of int, no SOS/EOT
        embs = tok_emb[toks]             # [T, 512]
        T = embs.shape[0]
        if T >= n_ctx:
            # evenly sample n_ctx tokens
            idx = torch.linspace(0, T - 1, n_ctx).long()
            out[k] = embs[idx]
        else:
            out[k, :T] = embs
            out[k, T:] = embs.mean(0, keepdim=True).expand(n_ctx - T, -1)
    return out


class TextAnchorModule(nn.Module):
    """
    K learnable text context vectors a frozen CLIP text encoder a linear projection
    a K anchor vectors [K, backbone_dim=768].

    Patch tokens (768-dim) cross-attend to these text-grounded anchors, giving
    the refinement module a semantic, language-driven set of reference points.

    The CLIP text encoder is always frozen.  Only ctx_anchors and proj are trained.
    """
    def __init__(self, clipreid_model, K: int, backbone_dim: int = 768,
                 n_part_ctx: int = 0, clip_model=None):
        super().__init__()
        pl      = clipreid_model.prompt_learner
        ctx_dim = pl.ctx_generic.shape[2]    # 512

        # SOS prefix and "person." + EOT suffix (static CLIP embeddings)
        self.register_buffer("token_prefix",      pl.token_prefix[0].float().clone())   # [1,  512]
        self.register_buffer("token_suffix",      pl.token_suffix[0].float().clone())   # [S,  512]
        self.register_buffer("tokenized_prompts", pl.tokenized_prompts.clone())          # [1, seq]

        # Total context length = seq_len - prefix_len - suffix_len (must be 16 here)
        seq_len     = pl.tokenized_prompts.shape[1]          # 77
        prefix_len  = self.token_prefix.shape[0]             # 1
        suffix_len  = self.token_suffix.shape[0]             # e.g. 60
        n_total_ctx = seq_len - prefix_len - suffix_len      # 16

        n_part_ctx  = min(n_part_ctx, n_total_ctx)
        n_main_ctx  = n_total_ctx - n_part_ctx

        generic_len = pl.ctx_generic.shape[1]            # 8
        ctx_mean    = pl.ctx_generic.detach().float().mean(dim=0)   # [8, 512]

        # ctx_main: learnable, initialised from phrase token embeddings (or ctx_generic mean)
        ctx_main_init = _init_ctx_from_phrases(K, n_main_ctx, ctx_dim, ctx_mean, clip_model)
        if clip_model is None:
            # fallback: init from ctx_generic mean
            ctx_main_init[:, :min(generic_len, n_main_ctx)] = \
                ctx_mean[:min(generic_len, n_main_ctx)].unsqueeze(0).expand(K, -1, -1)
        self.ctx_main = nn.Parameter(ctx_main_init + 0.01 * torch.randn_like(ctx_main_init))

        # ctx_parts: frozen buffer initialised from phrase token embeddings
        ctx_parts_init = _init_ctx_from_phrases(K, n_part_ctx, ctx_dim, ctx_mean, clip_model)
        self.register_buffer("ctx_parts", ctx_parts_init)

        # Frozen text encoder (trained in Stage 1/2 a keep frozen here)
        self.text_encoder = clipreid_model.text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # 512 a backbone_dim (768) projection a the key cross-modal bridge
        self.proj = nn.Linear(ctx_dim, backbone_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

        print(f"  TextAnchorModule: K={K}  n_main_ctx={n_main_ctx}  n_part_ctx={n_part_ctx}"
              f"  ({'phrase init' if clip_model is not None else 'ctx_generic init'})")

    def forward(self) -> torch.Tensor:
        """Returns [K, backbone_dim] anchor vectors."""
        K     = self.ctx_main.shape[0]
        dtype = next(self.text_encoder.parameters()).dtype

        prefix  = self.token_prefix.unsqueeze(0).expand(K, -1, -1).to(dtype)
        suffix  = self.token_suffix.unsqueeze(0).expand(K, -1, -1).to(dtype)
        ctx     = torch.cat([self.ctx_main, self.ctx_parts], dim=1).to(dtype)  # [K, 16, 512]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)                       # [K, seq, 512]

        tok        = self.tokenized_prompts.expand(K, -1)
        text_feats = self.text_encoder(prompts, tok)                            # [K, 512]
        return self.proj(text_feats.float())                                    # [K, backbone_dim]


class FreeAnchorModule(nn.Module):
    """K unconstrained learnable anchor vectors with no text grounding.

    Replaces TextAnchorModule when ``use_free_anchors: true`` is set in config.
    Anchors are randomly initialised and learned end-to-end alongside the rest
    of the visual module, with no language prior.  Useful as an ablation to
    measure the contribution of semantic grounding.

    Has the same zero-argument call interface as TextAnchorModule:
        anchors = self.text_anchor()  a  [K, backbone_dim]
    """

    def __init__(self, K: int, backbone_dim: int = 768):
        super().__init__()
        self.anchors = nn.Parameter(torch.empty(K, backbone_dim))
        nn.init.trunc_normal_(self.anchors, std=0.02)

    def forward(self) -> torch.Tensor:
        """Returns [K, backbone_dim] anchor vectors."""
        return self.anchors


class CLIPReIDWithVisual(nn.Module):
    """Frozen CLIP-ReID backbone + text-driven SemanticRefinementModule + classifier."""

    def __init__(self, backbone, clipreid_model, num_classes: int, raw_cfg: Dict,
                 camera_num: int = 0, clip_model=None):
        super().__init__()
        self.backbone       = backbone        # image_encoder from build_transformer
        self.clipreid_model = clipreid_model  # full model a used for I2T text features

        D         = 768            # ViT-B/16 hidden dim
        K         = raw_cfg.get("s3_num_anchors",        6)
        M         = raw_cfg.get("s3_num_domain_anchors", 3)
        n_layers  = raw_cfg.get("s3_refine_layers",      1)
        embed_dim = raw_cfg.get("s3_embed_dim",          768)

        # Which transformer block(s) to extract patch tokens from.
        # s3_token_layers (list) takes priority over s3_token_layer (int).
        _tl_list = raw_cfg.get("s3_token_layers", None)
        if _tl_list is not None:
            self.token_layer = list(_tl_list)   # e.g. [6, 9]
        else:
            self.token_layer = raw_cfg.get("s3_token_layer", None)  # int or None


        # Anchor vectors a text-grounded (default) or unconstrained learnable.
        if raw_cfg.get("use_free_anchors", False):
            self.text_anchor = FreeAnchorModule(K, backbone_dim=D)
            print(f"  Semantic anchors: free (unconstrained learnable, K={K})")
        else:
            n_part = raw_cfg.get("s3_anchor_part_ctx", 0)
            self.text_anchor = TextAnchorModule(clipreid_model, K, backbone_dim=D,
                                                n_part_ctx=n_part, clip_model=clip_model)
            print(f"  Semantic anchors: text-grounded (K={K})")

        # Dynamic per-image domain anchors a set s3_num_domain_anchors=0 to disable.
        use_cam_anchors = raw_cfg.get("use_sie_camera", False) and camera_num > 0
        if M == 0:
            self.domain_anchor = None
        elif use_cam_anchors:
            self.domain_anchor = CameraConditionedDomainAnchorGenerator(D, M, camera_num)
            print(f"  Domain anchors: camera-conditioned  ({camera_num} cameras, M={M})")
        elif raw_cfg.get("use_clip_domain_anchors", False):
            self.domain_anchor = CLIPGroundedDomainAnchorGenerator(D, M)
        else:
            self.domain_anchor = DomainAnchorGenerator(D, M)

        # Cross-attention refinement: patch tokens attend to text + domain anchors.
        # When s3_token_layers specifies multiple layers, one SemanticRefinementModule
        # per path with separate weights.  Anchors are partitioned:
        #   path 0 (earlier layer): text_anchors[:K//2]
        #   path 1 (deeper  layer): text_anchors[K//2:] + all domain_anchors
        # Path outputs are summed (not concatenated) so proj dimension is unchanged.
        dropout = raw_cfg.get("s3_dropout", 0.0)
        if isinstance(self.token_layer, list):
            self.refine = nn.ModuleList([
                SemanticRefinementModule(D, num_heads=8, num_layers=n_layers, dropout=dropout)
                for _ in self.token_layer
            ])
        else:
            self.refine = SemanticRefinementModule(D, num_heads=8, num_layers=n_layers, dropout=dropout)

        # Projection + BN + classifiers (dimension unchanged for both single and multi-path)
        self.proj       = nn.Linear(D, embed_dim) if embed_dim != D else nn.Identity()
        self.bn         = nn.BatchNorm1d(embed_dim)
        self.bn.bias.requires_grad_(False)
        n_classifiers   = raw_cfg.get("s3_num_classifiers", 1)
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes, bias=False) for _ in range(n_classifiers)
        ])
        self.classifier_dropout = raw_cfg.get("s3_classifier_dropout", 0.0)

        # Projection into CLIP text space (512-dim) for I2T regularization on refined feat
        self.text_proj  = nn.Linear(embed_dim, 512, bias=False)

        for cls in self.classifiers:
            nn.init.normal_(cls.weight, std=0.001)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias,   0.0)
        nn.init.xavier_uniform_(self.text_proj.weight)

        # Cache for part eval (set in encode(), consumed by encode_eval_part())
        self._cached_attn_w:         Optional[torch.Tensor] = None
        self._cached_refined_tokens: Optional[torch.Tensor] = None
        self._cached_content_tokens: Optional[torch.Tensor] = None  # kept for compat
        self._cached_K:              int = raw_cfg.get("s3_num_anchors", 6)

    @torch.no_grad()
    def extract_tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run frozen backbone; return patch tokens [B, N, 768] fp32."""
        feat_last, _, _ = self.backbone(imgs, token_layer=self.token_layer)
        return feat_last[:, 1:].float()   # exclude CLS token

    def extract_tokens_and_proj(self, imgs: torch.Tensor):
        """Run frozen backbone; return (patch_tokens [B, N, 768], img_feat [B, 768], img_proj [B, 512]) fp32."""
        feat_last, img_feat_seq, img_feat_proj_seq = self.backbone(imgs, token_layer=self.token_layer)
        return feat_last[:, 1:].float(), img_feat_seq[:, 0].float(), img_feat_proj_seq[:, 0].float()

    def _get_domain_anchors(self, tokens, clip_proj, cam_ids, text_anch):
        """Dispatch to the right domain anchor generator."""
        if self.domain_anchor is None:
            return None
        if isinstance(self.domain_anchor, CameraConditionedDomainAnchorGenerator):
            if cam_ids is None:
                # No camera label a fallback: use mean camera embedding alongside visual tokens
                B = tokens.shape[0]
                mean_cam = self.domain_anchor.cam_embed.weight.mean(0, keepdim=True).expand(B, -1)
                visual   = tokens.mean(dim=1)
                combined = torch.cat([visual, mean_cam], dim=1)
                return self.domain_anchor.anchor_gen(combined).view(
                    B, self.domain_anchor.num_anchors, self.domain_anchor.dim)
            return self.domain_anchor(tokens, cam_ids)
        if isinstance(self.domain_anchor, CLIPGroundedDomainAnchorGenerator):
            return self.domain_anchor(clip_proj, text_anch)
        return self.domain_anchor(tokens)   # DomainAnchorGenerator (visual mean-pool)

    def encode(self, tokens: torch.Tensor,
               clip_proj: Optional[torch.Tensor] = None,
               cam_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens: [B, N, D] patch tokens.  Returns embedding [B, embed_dim].

        Patch tokens cross-attend to text/domain anchors, then are pooled using
        attention-weighted pooling (tokens with stronger anchor alignment weighted higher).

        clip_proj: [B, 512] CLIP image projection a required when use_clip_domain_anchors=True.
        cam_ids:   [B] int64 camera indices    a required when use_sie_camera=True (Stage 3+).
        """
        B         = tokens.shape[0]
        text_anch = self.text_anchor()                           # [K, 768]
        text_anch = text_anch.unsqueeze(0).expand(B, -1, -1)    # [B, K, 768]

        if isinstance(self.refine, nn.ModuleList):
            # Partitioned-anchor parallel mode.
            n_paths    = len(self.refine)
            N          = tokens.shape[1] // n_paths
            tok_splits = list(tokens.split(N, dim=1))

            domain_anch = self._get_domain_anchors(tok_splits[-1], clip_proj, cam_ids, text_anch)

            K_half      = text_anch.shape[1] // 2
            anch_splits = [text_anch[:, :K_half, :]]
            deep_anch   = text_anch[:, K_half:, :] if domain_anch is None else \
                          torch.cat([text_anch[:, K_half:, :], domain_anch], dim=1)
            anch_splits.append(deep_anch)

            path_feats = []
            for refine_i, tok_i, anch_i in zip(self.refine, tok_splits, anch_splits):
                refined_i, attn_w_i = refine_i(tok_i, anch_i)
                w_i = attn_w_i.max(dim=-1).values
                w_i = w_i / w_i.sum(dim=1, keepdim=True).clamp(min=1e-6)
                path_feats.append((refined_i * w_i.unsqueeze(-1)).sum(dim=1))
            feat = self.proj(sum(path_feats))
        else:
            domain_anch = self._get_domain_anchors(tokens, clip_proj, cam_ids, text_anch)
            anchors     = text_anch if domain_anch is None else \
                          torch.cat([text_anch, domain_anch], dim=1)
            refined, attn_w = self.refine(tokens, anchors)
            self._cached_attn_w         = attn_w.detach()
            self._cached_refined_tokens = refined.detach()
            self._cached_content_tokens = tokens.detach()      # kept for compat
            self._cached_K              = text_anch.shape[1]   # identity anchors only
            token_weights   = attn_w.max(dim=-1).values
            token_weights   = token_weights / token_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            feat            = self.proj((refined * token_weights.unsqueeze(-1)).sum(dim=1))

        return feat

    def encode_with_attn(self, tokens: torch.Tensor,
                         clip_proj: Optional[torch.Tensor] = None,
                         cam_ids: Optional[torch.Tensor] = None):
        """Like encode() but also returns per-anchor attention weights.

        Returns:
            feat:    [B, embed_dim]
            attn_w:  [B, N, K+M]  a cross-attention weights from the final refine layer
                     (single-path only; multi-path returns list of per-path weights)
        """
        B         = tokens.shape[0]
        text_anch = self.text_anchor()
        text_anch = text_anch.unsqueeze(0).expand(B, -1, -1)

        if isinstance(self.refine, nn.ModuleList):
            n_paths    = len(self.refine)
            N          = tokens.shape[1] // n_paths
            tok_splits = list(tokens.split(N, dim=1))
            domain_anch = self._get_domain_anchors(tok_splits[-1], clip_proj, cam_ids, text_anch)
            K_half      = text_anch.shape[1] // 2
            anch_splits = [text_anch[:, :K_half, :]]
            deep_anch   = text_anch[:, K_half:, :] if domain_anch is None else \
                          torch.cat([text_anch[:, K_half:, :], domain_anch], dim=1)
            anch_splits.append(deep_anch)
            path_feats, attn_ws = [], []
            for refine_i, tok_i, anch_i in zip(self.refine, tok_splits, anch_splits):
                refined_i, attn_w_i = refine_i(tok_i, anch_i)
                w_i = attn_w_i.max(dim=-1).values
                w_i = w_i / w_i.sum(dim=1, keepdim=True).clamp(min=1e-6)
                path_feats.append((refined_i * w_i.unsqueeze(-1)).sum(dim=1))
                attn_ws.append(attn_w_i)
            feat   = self.proj(sum(path_feats))
            attn_w = attn_ws
        else:
            domain_anch = self._get_domain_anchors(tokens, clip_proj, cam_ids, text_anch)
            anchors     = text_anch if domain_anch is None else \
                          torch.cat([text_anch, domain_anch], dim=1)
            refined, attn_w = self.refine(tokens, anchors)
            token_weights   = attn_w.max(dim=-1).values
            token_weights   = token_weights / token_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            feat            = self.proj((refined * token_weights.unsqueeze(-1)).sum(dim=1))

        return feat, attn_w

    def forward(self, tokens: torch.Tensor,
                clip_proj: Optional[torch.Tensor] = None,
                cam_ids: Optional[torch.Tensor] = None):
        feat     = self.encode(tokens, clip_proj=clip_proj, cam_ids=cam_ids)
        feat_bn  = self.bn(feat)
        logit    = [cls(F.dropout(feat_bn, p=self.classifier_dropout, training=self.training))
                    for cls in self.classifiers]
        return logit, feat

    def forward_img(self, imgs: torch.Tensor,
                    cam_ids: Optional[torch.Tensor] = None):
        """End-to-end forward: images a backbone (trainable) a visual module.
        Returns (logit, feat, img_feature, img_feat_proj) where:
          img_feature:   [B, 768] x12 CLS a for S2-style id/triplet loss
          img_feat_proj: [B, 512] CLIP-projected CLS a for I2T loss
        """
        feat_last, img_feat_seq, img_feat_proj_seq = self.backbone(imgs, token_layer=self.token_layer)
        tokens        = feat_last[:, 1:].float()          # [B, N, 768] patch tokens
        img_feature   = img_feat_seq[:, 0].float()        # [B, 768] x12 CLS
        img_feat_proj = img_feat_proj_seq[:, 0].float()   # [B, 512]
        logit, feat   = self.forward(tokens, clip_proj=img_feat_proj, cam_ids=cam_ids)
        return logit, feat, img_feature, img_feat_proj

    def forward_img_with_parts(self, imgs: torch.Tensor,
                                cam_ids: Optional[torch.Tensor] = None):
        """Like forward_img() but also returns per-anchor features for consistency loss."""
        feat_last, img_feat_seq, img_feat_proj_seq = self.backbone(imgs, token_layer=self.token_layer)
        tokens        = feat_last[:, 1:].float()
        img_feature   = img_feat_seq[:, 0].float()
        img_feat_proj = img_feat_proj_seq[:, 0].float()
        logit, feat, per_anchor, attn_mass = self.forward_with_parts(
            tokens, clip_proj=img_feat_proj, cam_ids=cam_ids)
        return logit, feat, img_feature, img_feat_proj, per_anchor, attn_mass

    def encode_eval(self, imgs: torch.Tensor,
                    cam_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Eval feature: refined_feat(768) | img_feature(768) | img_proj(512) = 2048-dim."""
        feat_last, img_feat_seq, img_proj_seq = self.backbone(imgs, token_layer=self.token_layer)
        tokens        = feat_last[:, 1:].float()         # [B, N, 768] patch tokens
        img_feature   = img_feat_seq[:, 0].float()       # [B, 768] x12 CLS
        img_feat_proj = img_proj_seq[:, 0].float()       # [B, 512] CLIP-projected CLS
        refined_feat  = self.encode(tokens, clip_proj=img_feat_proj, cam_ids=cam_ids)  # [B, 768]
        return torch.cat([refined_feat, img_feature, img_feat_proj], dim=1)  # [B, 2048]

    @torch.no_grad()
    def encode_eval_part(self, top_k: int = 5) -> torch.Tensor:
        """Per-anchor part features a call immediately after encode_eval().

        Reuses attn_w and refined tokens cached by the preceding encode_eval() call
        so no extra backbone pass is needed.  Single-path mode only.

        For each of the K_id identity anchors: attention-weighted pool of *refined*
        tokens a project (same self.proj as global feat) a per-anchor L2-normalise
        a zero out non-top-K slots by attention mass.

        Returns [B, K_id * embed_dim].
        """
        attn_w  = self._cached_attn_w
        refined = self._cached_refined_tokens
        if attn_w is None or refined is None:
            raise RuntimeError(
                "encode_eval_part must be called immediately after encode_eval().")

        B    = refined.shape[0]
        K_id = self._cached_K

        attn_id    = attn_w[:, :, :K_id]                          # [B, N, K_id]
        per_anchor = attn_id.transpose(-2, -1) @ refined           # [B, K_id, D]
        # Project into embed_dim (same as global feature path)
        D_in = per_anchor.shape[-1]
        per_anchor = self.proj(per_anchor.reshape(B * K_id, D_in)) # [B*K_id, embed_dim]
        per_anchor = per_anchor.reshape(B, K_id, -1)               # [B, K_id, embed_dim]
        per_anchor = F.normalize(per_anchor, p=2, dim=-1)          # per-anchor L2-norm

        if top_k < K_id:
            attn_mass = attn_id.sum(dim=1)                         # [B, K_id]
            threshold = attn_mass.topk(k=top_k, dim=1).values[:, -1:]
            mask      = (attn_mass >= threshold).float().unsqueeze(-1)
            per_anchor = per_anchor * mask

        return per_anchor.reshape(B, -1)                           # [B, K_id * embed_dim]

    def forward_with_parts(self, tokens: torch.Tensor,
                           clip_proj: Optional[torch.Tensor] = None,
                           cam_ids: Optional[torch.Tensor] = None):
        """Like forward() but also returns per-anchor features for consistency loss.

        Only supports single-path mode (when self.refine is a single module).
        Falls back to regular forward() in multi-path mode (returns None, None).

        Returns
        -------
        logit      : [B, num_classes]
        feat       : [B, embed_dim]
        per_anchor : [B, K_id, embed_dim]  per-anchor L2-normalised projected features (or None)
        attn_mass  : [B, K_id]            sum of attention weights per anchor (or None)
        """
        if isinstance(self.refine, nn.ModuleList):
            logit, feat = self.forward(tokens, clip_proj=clip_proj, cam_ids=cam_ids)
            return logit, feat, None, None

        B         = tokens.shape[0]
        text_anch = self.text_anchor()
        text_anch = text_anch.unsqueeze(0).expand(B, -1, -1)

        domain_anch = self._get_domain_anchors(tokens, clip_proj, cam_ids, text_anch)
        anchors     = text_anch if domain_anch is None else \
                      torch.cat([text_anch, domain_anch], dim=1)
        refined, attn_w = self.refine(tokens, anchors)

        K_id          = text_anch.shape[1]
        token_weights = attn_w.max(dim=-1).values
        token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        feat          = self.proj((refined * token_weights.unsqueeze(-1)).sum(dim=1))
        feat_bn       = self.bn(feat)
        logit         = [cls(feat_bn) for cls in self.classifiers]

        # Per-anchor features: pool refined tokens, project a same path as global feat
        attn_id    = attn_w[:, :, :K_id]                              # [B, N, K_id]
        per_anchor = attn_id.transpose(-2, -1) @ refined               # [B, K_id, D]
        D_in       = per_anchor.shape[-1]
        per_anchor = self.proj(per_anchor.reshape(B * K_id, D_in))     # [B*K_id, embed_dim]
        per_anchor = per_anchor.reshape(B, K_id, -1)                   # [B, K_id, embed_dim]
        per_anchor = F.normalize(per_anchor, p=2, dim=-1)
        attn_mass  = attn_id.sum(dim=1)                                # [B, K_id]

        return logit, feat, per_anchor, attn_mass


def _anchor_triplet_loss(per_anchor: torch.Tensor,
                         attn_mass:  torch.Tensor,
                         target:     torch.Tensor,
                         margin:     float = 0.3) -> torch.Tensor:
    """Batch-hard triplet loss applied independently per anchor, weighted by aliveness.

    per_anchor : [B, K_id, embed_dim]  per-anchor L2-normalised projected features
    attn_mass  : [B, K_id]     attention mass (a patch weights) per anchor a measures
                                how "alive" each anchor is in each image
    target     : [B]           person IDs
    margin     : float         triplet margin (cosine space)

    For each anchor slot k, only images where that anchor is active (high attn_mass)
    contribute to the loss.  Aliveness is normalised per-anchor across the batch so
    the most active image for anchor k always has weight 1.

    Loss formulation (cosine similarity a higher = more similar):
        L_k = relu(margin - (pos_sim_k - neg_sim_k))
        where pos_sim_k = hardest positive (min same-ID sim, excl. self)
              neg_sim_k = hardest negative (max diff-ID sim)
    """
    B, K_id, D = per_anchor.shape
    device = per_anchor.device

    # Pairwise cosine similarity for each anchor: [K_id, B, B]
    pa_t = per_anchor.permute(1, 0, 2)                        # [K_id, B, D]
    sim  = torch.bmm(pa_t, pa_t.transpose(-2, -1))            # [K_id, B, B]

    same_pid = (target.unsqueeze(1) == target.unsqueeze(0))   # [B, B]
    diff_pid = ~same_pid
    diag     = torch.eye(B, dtype=torch.bool, device=device)

    pos_mask = (same_pid & ~diag).unsqueeze(0).expand(K_id, -1, -1)   # [K_id, B, B]
    neg_mask = diff_pid.unsqueeze(0).expand(K_id, -1, -1)

    INF = 1e4
    # Hardest positive per anchor per image (min same-ID cosine similarity)
    pos_sim = (sim * pos_mask.float() +
               (~pos_mask).float() * INF).min(dim=-1).values          # [K_id, B]
    # Hardest negative (max diff-ID cosine similarity)
    neg_sim = (sim * neg_mask.float() +
               (~neg_mask).float() * (-INF)).max(dim=-1).values        # [K_id, B]

    raw_loss = torch.clamp(margin - (pos_sim - neg_sim), min=0.0)      # [K_id, B]

    # Aliveness gate: [K_id, B], normalise per anchor so max = 1
    alive = attn_mass.t()                                               # [K_id, B]
    alive = alive / alive.amax(dim=1, keepdim=True).clamp(min=1e-6)

    # Only penalise samples that have at least one positive in the batch
    has_pos = same_pid.any(dim=1).unsqueeze(0).expand(K_id, -1)        # [K_id, B]
    weights = alive * has_pos.float()

    denom = weights.sum().clamp(min=1e-6)
    return (raw_loss * weights).sum() / denom


@torch.no_grad()
def _validate_stage3(vis_model, val_loader, num_query, device,
                     rerank: bool = False) -> Tuple[float, float]:
    vis_model.eval()
    eval1 = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes", reranking=False)
    eval1.reset()
    if rerank:
        eval2 = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes", reranking=True)
        eval2.reset()
    for imgs, pids, camids, camids_batch, viewids, _ in val_loader:
        imgs = imgs.to(device)
        feat = vis_model.encode_eval(imgs, cam_ids=camids_batch.to(device))
        feat = F.normalize(feat, p=2, dim=1)
        eval1.update((feat.cpu(), pids, camids))
        if rerank:
            eval2.update((feat.cpu(), pids, camids))
    cmc, mAP, *_ = eval1.compute()
    if rerank:
        cmc2, mAP2, *_ = eval2.compute()
        print(f"  [rerank] mAP: {float(mAP2)*100:.1f}%  Rank-1: {float(cmc2[0])*100:.1f}%", end="")
    return float(cmc[0]) * 100, float(mAP) * 100


def _run_s3_phase(vis_model, cache_loader, val_loader, num_query,
                  num_classes: int, raw_cfg: Dict, device, ckpt_dir: str,
                  epochs: int, base_lr: float, warmup_ep: int,
                  save_period: int, phase: str, best_mAP: float,
                  epoch_offset: int = 0, decay: bool = False,
                  text_feats_all: torch.Tensor = None,
                  img_loader=None, backbone_lr: float = None,
                  use_i2t: bool = False,
                  use_i2t_refined: bool = False,
                  use_s2_loss: bool = False,
                  refine_lr: float = None) -> float:
    """Single training loop shared by Stage 3a and 3b.

    When img_loader and backbone_lr are provided the backbone is unfrozen and
    trained at backbone_lr while visual-module params are trained at base_lr.
    In that case raw images are used directly (cache is bypassed).
    use_i2t_refined: I2T loss on refined feat projected into CLIP text space (text_proj).
    use_s2_loss: also supervise on x12 CLS (img_feature) with S2-style id+triplet losses.
    refine_lr: if set, the refine module uses this LR instead of base_lr.
    """
    use_images         = img_loader is not None and backbone_lr is not None
    use_images_frozen  = img_loader is not None and backbone_lr is None
    lambda_div         = raw_cfg.get("lambda_div",      0.0)
    lambda_cls_div     = raw_cfg.get("lambda_cls_div",  0.0)
    lambda_anc_cons    = raw_cfg.get("lambda_anc_cons", 0.0)
    cls_lr_scale       = raw_cfg.get("s3_classifier_lr_scale", 1.0)
    refine_params      = list(vis_model.refine.parameters())
    refine_param_ids   = {id(p) for p in refine_params}
    classifier_params  = list(vis_model.classifiers.parameters())
    classifier_ids     = {id(p) for p in classifier_params}

    if use_images:
        # Unfreeze backbone; visual modules should already have requires_grad=True
        for p in vis_model.backbone.parameters():
            p.requires_grad_(True)
        backbone_params = list(vis_model.backbone.parameters())
        backbone_ids    = {id(p) for p in backbone_params}
        head_params     = [p for p in vis_model.parameters()
                           if p.requires_grad
                           and id(p) not in backbone_ids
                           and id(p) not in refine_param_ids
                           and id(p) not in classifier_ids]
        param_groups = [
            {"params": backbone_params,   "lr": backbone_lr,              "name": "backbone"},
            {"params": head_params,       "lr": base_lr,                  "name": "head"},
            {"params": refine_params,     "lr": refine_lr or base_lr,     "name": "refine"},
            {"params": classifier_params, "lr": base_lr * cls_lr_scale,   "name": "classifier"},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=raw_cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda")
    else:
        other_params = [p for p in vis_model.parameters()
                        if p.requires_grad
                        and id(p) not in refine_param_ids
                        and id(p) not in classifier_ids]
        if refine_lr is not None or cls_lr_scale != 1.0:
            param_groups = [
                {"params": other_params,      "lr": base_lr,               "name": "head"},
                {"params": refine_params,     "lr": refine_lr or base_lr,  "name": "refine"},
                {"params": classifier_params, "lr": base_lr * cls_lr_scale,"name": "classifier"},
            ]
        else:
            param_groups = [{"params": other_params + refine_params + classifier_params,
                             "lr": base_lr, "name": "head"}]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=raw_cfg["weight_decay"])
        scaler = None

    min_lr_ratio = raw_cfg.get("s3_min_lr_ratio", 0.01)

    def _lr_lambda(ep):
        if ep < warmup_ep:
            return 0.1 + 0.9 * ep / max(warmup_ep, 1)
        if decay:
            t = (ep - warmup_ep) / max(epochs - warmup_ep, 1)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    xent    = CrossEntropyLabelSmooth(num_classes)
    def _xent_multi(logit, target):
        if isinstance(logit, list):
            return sum(xent(l, target) for l in logit) / len(logit)
        return xent(logit, target)
    triplet = TripletLoss(raw_cfg["margin"])

    for epoch in range(1, epochs + 1):
        vis_model.train()
        total_loss, t0 = 0.0, time.time()
        total_refine_xent = total_refine_tri = total_s2_xent = total_s2_tri = total_i2t = total_i2t_r = total_div = total_anc_cons = 0.0

        if use_images or use_images_frozen:
            loader_iter = img_loader
        else:
            loader_iter = cache_loader

        for batch in loader_iter:
            if use_images:
                imgs, pids, camids, *_ = batch
                imgs   = imgs.to(device)
                target = pids.to(device)
                cam_ids_d = camids.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    if lambda_anc_cons > 0:
                        logit, feat, img_feature, img_feat_proj, _pa, _am = \
                            vis_model.forward_img_with_parts(imgs, cam_ids=cam_ids_d)
                    else:
                        logit, feat, img_feature, img_feat_proj = \
                            vis_model.forward_img(imgs, cam_ids=cam_ids_d)
                        _pa, _am = None, None
                    l_rxent = _xent_multi(logit, target)
                    l_rtri  = raw_cfg["lambda_triplet"] * triplet(feat, target)[0]
                    loss = l_rxent + l_rtri
                    l_sxent = l_stri = l_i2t = l_i2t_r = l_anc = l_cls_div = 0.0
                    if lambda_anc_cons > 0 and _pa is not None:
                        l_anc = lambda_anc_cons * _anchor_triplet_loss(
                            _pa, _am, target, raw_cfg["margin"])
                        loss = loss + l_anc
                    if use_s2_loss:
                        s2_bn     = vis_model.clipreid_model.bottleneck(img_feature)
                        s2_score  = vis_model.clipreid_model.classifier(s2_bn)
                        l_sxent = raw_cfg["lambda_id"]      * xent(s2_score, target)
                        l_stri  = raw_cfg["lambda_triplet"] * triplet(img_feature, target)[0]
                        loss = loss + l_sxent + l_stri
                    if use_i2t and text_feats_all is not None:
                        l_i2t = raw_cfg["lambda_i2t"] * xent(
                            img_feat_proj @ text_feats_all.t(), target)
                        loss = loss + l_i2t
                    if use_i2t_refined and text_feats_all is not None:
                        feat_in_text = F.normalize(vis_model.text_proj(feat), dim=1)
                        l_i2t_r = 0.1 * raw_cfg["lambda_i2t"] * xent(
                            feat_in_text @ text_feats_all.t(), target)
                        loss = loss + l_i2t_r
                    if lambda_div > 0:
                        anch   = vis_model.text_anchor()
                        anch_n = F.normalize(anch, dim=-1)
                        G      = anch_n @ anch_n.T
                        mask   = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
                        l_div  = lambda_div * G[mask].pow(2).mean()
                        loss   = loss + l_div
                    if lambda_cls_div > 0 and len(vis_model.classifiers) > 1:
                        ws_f     = F.normalize(torch.stack(
                            [c.weight.flatten() for c in vis_model.classifiers]), dim=-1)
                        G_c      = ws_f @ ws_f.T
                        mask_c   = ~torch.eye(G_c.shape[0], dtype=torch.bool, device=G_c.device)
                        l_cls_div = lambda_cls_div * G_c[mask_c].pow(2).mean()
                        loss      = loss + l_cls_div
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_images_frozen:
                imgs, pids, camids, *_ = batch
                imgs      = imgs.to(device)
                target    = pids.to(device)
                cam_ids_d = camids.to(device)
                with torch.no_grad():
                    tokens_batch, img_feature_frozen, proj_batch_d = vis_model.extract_tokens_and_proj(imgs)
                optimizer.zero_grad()
                if lambda_anc_cons > 0:
                    logit, feat, _pa, _am = vis_model.forward_with_parts(
                        tokens_batch, clip_proj=proj_batch_d, cam_ids=cam_ids_d)
                else:
                    logit, feat = vis_model(tokens_batch, clip_proj=proj_batch_d, cam_ids=cam_ids_d)
                    _pa, _am = None, None
                l_rxent = _xent_multi(logit, target)
                l_rtri  = raw_cfg["lambda_triplet"] * triplet(feat, target)[0]
                loss = l_rxent + l_rtri
                l_sxent = l_stri = l_i2t = l_i2t_r = l_anc = l_cls_div = 0.0
                if lambda_anc_cons > 0 and _pa is not None:
                    l_anc = lambda_anc_cons * _anchor_triplet_loss(
                        _pa, _am, target, raw_cfg["margin"])
                    loss = loss + l_anc
                if use_s2_loss:
                    s2_bn    = vis_model.clipreid_model.bottleneck(img_feature_frozen)
                    s2_score = vis_model.clipreid_model.classifier(s2_bn)
                    l_sxent = raw_cfg["lambda_id"]      * xent(s2_score, target)
                    l_stri  = raw_cfg["lambda_triplet"] * triplet(img_feature_frozen, target)[0]
                    loss = loss + l_sxent + l_stri
                if use_i2t and text_feats_all is not None:
                    l_i2t = raw_cfg["lambda_i2t"] * xent(
                        proj_batch_d @ text_feats_all.t(), target)
                    loss = loss + l_i2t
                if use_i2t_refined and text_feats_all is not None:
                    feat_in_text = F.normalize(vis_model.text_proj(feat), dim=1)
                    l_i2t_r = 0.1 * raw_cfg["lambda_i2t"] * xent(
                        feat_in_text @ text_feats_all.t(), target)
                    loss = loss + l_i2t_r
                if lambda_div > 0:
                    anch   = vis_model.text_anchor()
                    anch_n = F.normalize(anch, dim=-1)
                    G      = anch_n @ anch_n.T
                    mask   = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
                    l_div  = lambda_div * G[mask].pow(2).mean()
                    loss   = loss + l_div
                if lambda_cls_div > 0 and len(vis_model.classifiers) > 1:
                    ws_f      = F.normalize(torch.stack(
                        [c.weight.flatten() for c in vis_model.classifiers]), dim=-1)
                    G_c       = ws_f @ ws_f.T
                    mask_c    = ~torch.eye(G_c.shape[0], dtype=torch.bool, device=G_c.device)
                    l_cls_div = lambda_cls_div * G_c[mask_c].pow(2).mean()
                    loss      = loss + l_cls_div
                loss.backward()
                optimizer.step()
            else:
                tokens_batch, proj_batch, cls_batch, pids_batch, cam_ids_batch = batch
                tokens_batch = tokens_batch.to(device).float()
                proj_batch_d = proj_batch.to(device).float()
                target       = pids_batch.to(device)
                cam_ids_d    = cam_ids_batch.to(device) if cam_ids_batch[0] >= 0 else None
                optimizer.zero_grad()
                if lambda_anc_cons > 0:
                    logit, feat, _pa, _am = vis_model.forward_with_parts(
                        tokens_batch, clip_proj=proj_batch_d, cam_ids=cam_ids_d)
                else:
                    logit, feat = vis_model(tokens_batch, clip_proj=proj_batch_d, cam_ids=cam_ids_d)
                    _pa, _am = None, None
                l_rxent = _xent_multi(logit, target)
                l_rtri  = raw_cfg["lambda_triplet"] * triplet(feat, target)[0]
                loss = l_rxent + l_rtri
                l_sxent = l_stri = l_i2t = l_i2t_r = l_anc = l_cls_div = 0.0
                if lambda_anc_cons > 0 and _pa is not None:
                    l_anc = lambda_anc_cons * _anchor_triplet_loss(
                        _pa, _am, target, raw_cfg["margin"])
                    loss = loss + l_anc
                if use_s2_loss:
                    img_feature = cls_batch.to(device).float()
                    s2_bn    = vis_model.clipreid_model.bottleneck(img_feature)
                    s2_score = vis_model.clipreid_model.classifier(s2_bn)
                    l_sxent = raw_cfg["lambda_id"]      * xent(s2_score, target)
                    l_stri  = raw_cfg["lambda_triplet"] * triplet(img_feature, target)[0]
                    loss = loss + l_sxent + l_stri
                if use_i2t and text_feats_all is not None:
                    proj_f = proj_batch.to(device).float()
                    l_i2t = raw_cfg["lambda_i2t"] * xent(
                        proj_f @ text_feats_all.t(), target)
                    loss = loss + l_i2t
                if use_i2t_refined and text_feats_all is not None:
                    feat_in_text = F.normalize(vis_model.text_proj(feat), dim=1)
                    l_i2t_r = 0.1 * raw_cfg["lambda_i2t"] * xent(
                        feat_in_text @ text_feats_all.t(), target)
                    loss = loss + l_i2t_r
                if lambda_div > 0:
                    anch   = vis_model.text_anchor()
                    anch_n = F.normalize(anch, dim=-1)
                    G      = anch_n @ anch_n.T
                    mask   = ~torch.eye(G.shape[0], dtype=torch.bool, device=G.device)
                    l_div  = lambda_div * G[mask].pow(2).mean()
                    loss   = loss + l_div
                if lambda_cls_div > 0 and len(vis_model.classifiers) > 1:
                    ws_f      = F.normalize(torch.stack(
                        [c.weight.flatten() for c in vis_model.classifiers]), dim=-1)
                    G_c       = ws_f @ ws_f.T
                    mask_c    = ~torch.eye(G_c.shape[0], dtype=torch.bool, device=G_c.device)
                    l_cls_div = lambda_cls_div * G_c[mask_c].pow(2).mean()
                    loss      = loss + l_cls_div
                loss.backward()
                optimizer.step()
            total_loss        += loss.item()
            total_refine_xent += float(l_rxent)
            total_refine_tri  += float(l_rtri)
            total_s2_xent     += float(l_sxent)
            total_s2_tri      += float(l_stri)
            total_i2t         += float(l_i2t)
            total_i2t_r       += float(l_i2t_r)
            total_div         += (float(l_div) if lambda_div > 0 else 0.0) + float(l_cls_div)
            total_anc_cons    += float(l_anc) if lambda_anc_cons > 0 else 0.0

        scheduler.step(epoch)
        elapsed = time.time() - t0
        n = max(len(loader_iter), 1)
        avg = total_loss / n
        abs_ep = epoch_offset + epoch
        detail = (f"rx={total_refine_xent/n:.3f} rt={total_refine_tri/n:.3f}"
                  + (f" sx={total_s2_xent/n:.3f} st={total_s2_tri/n:.3f}" if use_s2_loss else "")
                  + (f" i2t={total_i2t/n:.3f}" if use_i2t else "")
                  + (f" i2tr={total_i2t_r/n:.3f}" if use_i2t_refined else "")
                  + (f" dv={total_div/n:.4f}" if (lambda_div > 0 or lambda_cls_div > 0) else "")
                  + (f" ac={total_anc_cons/n:.4f}" if lambda_anc_cons > 0 else ""))
        cls_sim_str = ""
        if len(vis_model.classifiers) > 1:
            with torch.no_grad():
                ws = [F.normalize(c.weight.data.flatten(), dim=0) for c in vis_model.classifiers]
                pairs = [(i, j) for i in range(len(ws)) for j in range(i + 1, len(ws))]
                mean_sim = sum((ws[i] * ws[j]).sum().item() for i, j in pairs) / len(pairs)
            cls_sim_str = f" cs={mean_sim:.3f}"
        print(f"  [{phase} {epoch:3d}/{epochs}]  loss: {avg:.4f}  {detail}{cls_sim_str}  ({elapsed:.0f}s)", end="")

        if epoch % save_period == 0 or epoch == epochs:
            is_final = (epoch == epochs)
            rerank   = is_final and raw_cfg.get("eval_rerank", False)
            r1, mAP = _validate_fusion(vis_model, val_loader, num_query, device, raw_cfg)
            print(f"  mAP: {mAP:.1f}%  Rank-1: {r1:.1f}%", end="")
            torch.save(vis_model.state_dict(),
                       os.path.join(ckpt_dir, f"model_stage3_ep{abs_ep:03d}.pth"))
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(vis_model.state_dict(),
                           os.path.join(ckpt_dir, "model_stage3_best.pth"))
                print("  <- best", end="")
        print()

    return best_mAP


def _init_refine_from_clip_block(cross_block, clip_block) -> None:
    """Initialise a CrossAttentionBlock from a CLIP ResidualAttentionBlock.

    Weight mapping (all shapes match for ViT-B/16 with D=768, 8 heads, MLPA4):
      clip.attn.in_proj_{weight,bias}   a cross.attn.in_proj_{weight,bias}
      clip.attn.out_proj.{weight,bias}  a cross.attn.out_proj.{weight,bias}
      clip.ln_1 (pre-attn norm)         a cross.norm_q  and  cross.norm_kv
      clip.ln_2 (pre-mlp norm)          a cross.norm_ff
      clip.mlp.c_fc                     a cross.ff[0]   (Linear Da4D)
      clip.mlp.c_proj                   a cross.ff[3]   (Linear 4DaD)

    batch_first=True vs LND format does not affect the weight tensors themselves.
    """
    with torch.no_grad():
        cross_block.attn.in_proj_weight.copy_(clip_block.attn.in_proj_weight)
        cross_block.attn.in_proj_bias.copy_(clip_block.attn.in_proj_bias)
        cross_block.attn.out_proj.weight.copy_(clip_block.attn.out_proj.weight)
        cross_block.attn.out_proj.bias.copy_(clip_block.attn.out_proj.bias)
        cross_block.norm_q.weight.copy_(clip_block.ln_1.weight)
        cross_block.norm_q.bias.copy_(clip_block.ln_1.bias)
        cross_block.norm_kv.weight.copy_(clip_block.ln_1.weight)
        cross_block.norm_kv.bias.copy_(clip_block.ln_1.bias)
        cross_block.norm_ff.weight.copy_(clip_block.ln_2.weight)
        cross_block.norm_ff.bias.copy_(clip_block.ln_2.bias)
        cross_block.ff[0].weight.copy_(clip_block.mlp.c_fc.weight)
        cross_block.ff[0].bias.copy_(clip_block.mlp.c_fc.bias)
        cross_block.ff[3].weight.copy_(clip_block.mlp.c_proj.weight)
        cross_block.ff[3].bias.copy_(clip_block.mlp.c_proj.bias)


def run_stage3(clipreid_model, stage2_loader, val_loader, num_query,
               num_classes: int, raw_cfg: Dict, device, ckpt_dir: str,
               skip_s3a: bool = False, cam_num: int = 0):
    s3a_epochs  = raw_cfg.get("s3a_epochs",  20)
    s3a_lr      = raw_cfg.get("s3a_lr",      3.5e-4)
    s3a_warmup  = raw_cfg.get("s3a_warmup",  5)
    s3b_epochs      = raw_cfg.get("s3b_epochs",       80)
    s3b_lr          = raw_cfg.get("s3b_lr",           1e-4)
    s3b_warmup      = raw_cfg.get("s3b_warmup",       5)
    s3b_backbone_lr = raw_cfg.get("s3b_backbone_lr",  None)  # None = keep backbone frozen
    s3b_refine_lr   = raw_cfg.get("s3b_refine_lr",    None)  # None = same as s3b_lr
    save_period     = raw_cfg.get("stage3_save_period", 10)

    s3b_mode = f"bb_lr={s3b_backbone_lr:.1e}" if s3b_backbone_lr else "backbone frozen"
    print(f"\n[Stage 3] visual module fine-tuning  "
          f"3a={s3a_epochs}ep@{s3a_lr:.1e}  3b={s3b_epochs}ep@{s3b_lr:.1e}  ({s3b_mode})")

    # Load CLIP for phrase-based anchor initialisation (only needed when n_part_ctx > 0)
    _clip_for_init = None
    if raw_cfg.get("s3_anchor_part_ctx", 0) > 0:
        img_h, img_w = raw_cfg.get("img_size", [256, 128])
        stride = raw_cfg.get("stride_size", [12, 12])[0]
        h_res = int((img_h - 16) // stride + 1)
        w_res = int((img_w - 16) // stride + 1)
        _clip_for_init = load_clip_to_cpu("ViT-B-16", h_res, w_res, stride)
        print(f"  Loaded CLIP for phrase init (n_part_ctx={raw_cfg['s3_anchor_part_ctx']})")

    # Build the visual wrapper around the frozen backbone
    vis_model = CLIPReIDWithVisual(clipreid_model.image_encoder, clipreid_model,
                                   num_classes, raw_cfg, camera_num=cam_num,
                                   clip_model=_clip_for_init).to(device)
    del _clip_for_init

    # Initialise refine module(s) from CLIP blocks at the corresponding token layers.
    all_blocks = clipreid_model.image_encoder.transformer.resblocks
    _tl_layers = raw_cfg.get("s3_token_layers", None)
    if _tl_layers is not None:
        # Parallel-path mode: each SRM initialised from its own token layer's block.
        for path_refine, tl in zip(vis_model.refine, sorted(_tl_layers)):
            _idx = max(1, min(tl, len(all_blocks))) - 1
            for layer in path_refine.layers:
                _init_refine_from_clip_block(layer, all_blocks[_idx])
        _n_layers = len(vis_model.refine[0].layers)
        print(f"  Refine module ({len(vis_model.refine)} paths x {_n_layers} layer(s)) "
              f"initialised from CLIP blocks {sorted(_tl_layers)}")
    else:
        _tl_single = raw_cfg.get("s3_token_layer", None)
        _init_1idx = len(all_blocks) if _tl_single is None else max(1, min(_tl_single, len(all_blocks)))
        init_block = all_blocks[_init_1idx - 1]
        for layer in vis_model.refine.layers:
            _init_refine_from_clip_block(layer, init_block)
        print(f"  Refine module ({len(vis_model.refine.layers)} layer(s)) initialised from CLIP block {_init_1idx}/{len(all_blocks)}")

    # Backbone and text encoder stay frozen throughout Stage 3
    for p in vis_model.backbone.parameters():
        p.requires_grad_(False)
    if hasattr(vis_model.text_anchor, "text_encoder"):
        for p in vis_model.text_anchor.text_encoder.parameters():
            p.requires_grad_(False)

    # Token cache is only needed when the backbone stays frozen (S3a, or S3b without backbone_lr).
    # Skip extraction entirely when jumping straight to image-based S3b, or when s3_no_cache=True.
    no_cache   = raw_cfg.get("s3_no_cache", False)
    need_cache = not no_cache and not (skip_s3a and s3b_backbone_lr is not None)

    if need_cache:
        # Pre-extract and cache backbone tokens with multiple augmented views.
        # Use a sequential loader (shuffle=False) so image order is stable across
        # view passes, allowing views to be correctly stacked per image.
        n_views    = raw_cfg.get("s3_num_views", 4)
        nw         = raw_cfg["num_workers"]
        seq_loader = DataLoader(
            stage2_loader.dataset, batch_size=raw_cfg["batch_size"],
            shuffle=False, num_workers=nw, collate_fn=_train_collate,
            pin_memory=True, persistent_workers=nw > 0,
        )

        print(f"  Pre-extracting backbone tokens ({n_views} augmented views) ...")
        all_pids = []
        all_camids_cache = []
        for _, pids, camids, *_ in seq_loader:
            all_pids.append(pids)
            all_camids_cache.append(camids)
        all_pids   = torch.cat(all_pids)           # [N]
        all_camids_cache = torch.cat(all_camids_cache)  # [N]

        view_list = []
        proj_list = []
        cls_list  = []
        vis_model.eval()
        with torch.no_grad():
            for v in range(n_views):
                view_tokens = []
                view_projs  = []
                view_cls    = []
                for imgs, *_ in seq_loader:
                    imgs = imgs.to(device)
                    tok, cls, proj = vis_model.extract_tokens_and_proj(imgs)
                    view_tokens.append(tok.cpu().half())
                    view_projs.append(proj.cpu().half())
                    view_cls.append(cls.cpu().half())
                view_list.append(torch.cat(view_tokens))   # [N, T, 768]
                proj_list.append(torch.cat(view_projs))    # [N, D_proj]
                cls_list.append(torch.cat(view_cls))       # [N, 768]

        all_tokens = torch.stack(view_list, dim=1)   # [N, V, T, 768]
        all_projs  = torch.stack(proj_list, dim=1)   # [N, V, D_proj]
        all_cls    = torch.stack(cls_list,  dim=1)   # [N, V, 768]
        print(f"  Cached {len(all_pids)} samples x {n_views} views  "
              f"({all_tokens.element_size() * all_tokens.nelement() / 1e9:.2f} GB)")

        cache_ds     = _TokenCache(all_tokens, all_pids, projs=all_projs, cls_feats=all_cls,
                                   cam_ids=all_camids_cache)
        dataset_list = [(None, int(all_pids[i]), 0, 0) for i in range(len(all_pids))]
        from datasets.sampler import RandomIdentitySampler as _RIS
        sampler = _RIS(dataset_list, raw_cfg["batch_size"], raw_cfg["num_instance"])
        cache_loader = DataLoader(cache_ds, batch_size=raw_cfg["batch_size"],
                                  sampler=sampler, num_workers=0, pin_memory=True)
    elif no_cache:
        print(f"  Skipping token cache (s3_no_cache=true - extracting tokens on-the-fly each batch)")
        cache_loader = None
    else:
        print(f"  Skipping token cache (--start_s3b with backbone unfrozen - using images directly)")
        cache_loader = None

    print(f"  Text anchors initialised from mean of ctx_generic (K={raw_cfg.get('s3_num_anchors',6)})")

    # Pre-compute text features a text encoder is frozen throughout Stage 3
    s2_batch = raw_cfg.get("stage1_batch", 64)
    text_feats_all = []
    vis_model.clipreid_model.eval()
    with torch.no_grad():
        for i in range(0, num_classes, s2_batch):
            l = torch.arange(i, min(i + s2_batch, num_classes), device=device)
            with torch.amp.autocast("cuda"):
                tf = vis_model.clipreid_model(label=l, get_text=True)
            text_feats_all.append(tf.cpu())
    text_feats_all = torch.cat(text_feats_all).float().to(device)

    best_mAP = 0.0

    s3a_ckpt = os.path.join(ckpt_dir, "model_stage3a_best.pth")

    if skip_s3a:
        # Load best S3a checkpoint and jump straight to S3b
        if not os.path.exists(s3a_ckpt):
            raise FileNotFoundError(f"Stage 3a checkpoint not found: {s3a_ckpt}")
        _load_checkpoint(vis_model, s3a_ckpt, device, "S3a")
        print(f"  Loaded Stage 3a checkpoint from {s3a_ckpt}, skipping S3a.")
    else:
        # ------------------------------------------------------------------
        # Stage 3a: warm-up a all visual modules trained at higher LR
        # ------------------------------------------------------------------
        print(f"\n  [Stage 3a] visual module warm-up ({s3a_epochs} epochs)")

        best_mAP = _run_s3_phase(
            vis_model, cache_loader, val_loader, num_query,
            num_classes, raw_cfg, device, ckpt_dir,
            epochs=s3a_epochs, base_lr=s3a_lr, warmup_ep=s3a_warmup,
            save_period=save_period, phase="S3a", best_mAP=best_mAP,
            epoch_offset=0, text_feats_all=text_feats_all,
            img_loader=stage2_loader if no_cache else None,
        )
        # Save S3a best separately so S3b can be restarted independently
        torch.save(vis_model.state_dict(), s3a_ckpt)
        print(f"  Saved Stage 3a best to {s3a_ckpt}")

    # ------------------------------------------------------------------
    # Stage 3b: fine-tune at lower LR with cosine decay
    # ------------------------------------------------------------------
    # At the S3aaS3b boundary: reset BN running stats and reinitialise the
    # classifier to a single fresh head (standard path).
    # Exception: if s3_num_classifiers > 1, keep the existing multi-classifier
    # heads and BN stats so that S3b continues refining the diversity signal
    # without a cold restart.
    _n_cls_cfg = raw_cfg.get("s3_num_classifiers", 1)
    if _n_cls_cfg > 1:
        print(f"  s3_num_classifiers={_n_cls_cfg}: skipping BN/classifier reinit at S3b boundary")
    else:
        vis_model.bn.reset_running_stats()
        _dev     = next(vis_model.parameters()).device
        _num_cls = vis_model.classifiers[0].weight.shape[0]
        _emb_dim = vis_model.classifiers[0].weight.shape[1]
        _new_cls = nn.Linear(_emb_dim, _num_cls, bias=False).to(_dev)
        nn.init.normal_(_new_cls.weight, std=0.001)
        vis_model.classifiers = nn.ModuleList([_new_cls])
        vis_model.classifier_dropout = 0.0
        print(f"  BN running stats reinitialised; classifier reset to 1 for S3b")

    refine_lr_str = f"  refine_lr={s3b_refine_lr:.1e}" if s3b_refine_lr else ""
    print(f"\n  [Stage 3b] visual module fine-tune ({s3b_epochs} epochs){refine_lr_str}")

    best_mAP = _run_s3_phase(
        vis_model, cache_loader, val_loader, num_query,
        num_classes, raw_cfg, device, ckpt_dir,
        epochs=s3b_epochs, base_lr=s3b_lr, warmup_ep=s3b_warmup,
        save_period=save_period, phase="S3b", best_mAP=best_mAP,
        epoch_offset=s3a_epochs, decay=True, text_feats_all=text_feats_all,
        img_loader=stage2_loader if (s3b_backbone_lr or no_cache) else None,
        backbone_lr=s3b_backbone_lr,
        use_i2t=False,
        use_i2t_refined=False,
        use_s2_loss=False,
        refine_lr=s3b_refine_lr,
    )

    print(f"[Stage 3] done.  Best mAP: {best_mAP:.1f}%")


# ===========================================================================
# Stage 4 a end-to-end fine-tune (backbone unfrozen + visual module)
# ===========================================================================
@torch.no_grad()
def _validate_stage4(vis_model, val_loader, num_query, device,
                     rerank: bool = False, tta: bool = False,
                     raw_cfg: Optional[Dict] = None) -> Tuple[float, float]:
    vis_model.eval()
    all_feats, all_pids, all_camids = [], [], []
    for imgs, pids, camids, camids_batch, viewids, img_paths in val_loader:
        imgs = imgs.to(device)
        cam_ids_d = camids_batch.to(device)
        feat = vis_model.encode_eval(imgs, cam_ids=cam_ids_d)
        if tta:
            feat_flip = vis_model.encode_eval(imgs.flip(dims=[3]), cam_ids=cam_ids_d)
            feat = (feat + feat_flip) * 0.5
        all_feats.append(feat.cpu())   # raw, un-normalised
        all_pids.append(torch.as_tensor(pids))
        all_camids.append(torch.as_tensor(camids))

    feats  = torch.cat(all_feats)   # raw [N, D]
    pids   = torch.cat(all_pids)
    camids = torch.cat(all_camids)

    # Normalise features before metric computation.
    feats = F.normalize(feats, p=2, dim=1)

    # NFC after normalisation (Pose2ID order)
    if raw_cfg and raw_cfg.get("use_nfc", False):
        k1, k2 = raw_cfg.get("nfc_k1", 2), raw_cfg.get("nfc_k2", 2)
        print(f"  [NFC] k1={k1}  k2={k2}", end="")
        feats = apply_nfc_split(feats, num_query, k1=k1, k2=k2)

    eval1 = R1_mAP_eval(num_query, max_rank=50, feat_norm="no", reranking=False)
    eval1.reset()
    eval1.update((feats, pids, camids))
    cmc, mAP, *_ = eval1.compute()
    if rerank:
        eval2 = R1_mAP_eval(num_query, max_rank=50, feat_norm="no", reranking=True)
        eval2.reset()
        eval2.update((feats, pids, camids))
        cmc2, mAP2, *_ = eval2.compute()
        print(f"  [rerank] mAP: {float(mAP2)*100:.1f}%  Rank-1: {float(cmc2[0])*100:.1f}%", end="")
    return float(cmc[0]) * 100, float(mAP) * 100


@torch.no_grad()
def _validate_fusion(vis_model, val_loader, num_query, device,
                     raw_cfg: Dict) -> Tuple[float, float]:
    """Score-level fusion: evaluate each feature component independently, then
    report a weighted cosine-similarity fusion.

    Prints mAP / Rank-1 for:
      - refined_feat  (embed_dim-d)
      - img_feature   (768-d backbone CLS)
      - img_feat_proj (512-d CLIP projection)
      - concat        (2048-d, same as standard eval)
      - fusion        (weighted cosine sim, configurable weights)

    Config keys:
      fusion_w_refined  (default 1.0)
      fusion_w_imgfeat  (default 1.0)
      fusion_w_proj     (default 1.0)
    """
    w_r = raw_cfg.get("fusion_w_refined", 1.0)
    w_i = raw_cfg.get("fusion_w_imgfeat", 1.0)
    w_p = raw_cfg.get("fusion_w_proj",    1.0)
    embed_dim   = raw_cfg.get("s3_embed_dim", 768)  # refined feat dim
    backbone_dim = 768                               # ViT CLS dim (constant)

    vis_model.eval()
    part_k       = raw_cfg.get("s3_eval_part_k", 0)
    has_part_eval = part_k > 0 and hasattr(vis_model, "encode_eval_part")

    all_feats, all_pids, all_camids = [], [], []
    all_part_feats     = [] if has_part_eval else None
    all_part_feats_all = [] if has_part_eval else None  # all anchors, no top-K mask
    n_batches = len(val_loader)
    print(f"  Extracting features: {n_batches} batches  "
          f"({len(val_loader.dataset)} images  num_query={num_query})")
    for batch_idx, (imgs, pids, camids, camids_batch, viewids, img_paths) in enumerate(val_loader):
        imgs  = imgs.to(device)
        cam_d = camids_batch.to(device)
        all_feats.append(vis_model.encode_eval(imgs, cam_ids=cam_d).cpu())
        if has_part_eval:
            # encode_eval_part reuses attn_w cached by the preceding encode_eval call a
            # no extra backbone pass.
            all_part_feats.append(vis_model.encode_eval_part(top_k=part_k).cpu())
            K_id_now = vis_model._cached_K
            if part_k < K_id_now:
                all_part_feats_all.append(vis_model.encode_eval_part(top_k=K_id_now).cpu())
        all_pids.append(torch.as_tensor(pids))
        all_camids.append(torch.as_tensor(camids))
        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == 0:
            print(f"    batch {batch_idx+1}/{n_batches}", flush=True)

    feats  = torch.cat(all_feats)    # raw [N, D]
    pids   = torch.cat(all_pids)
    camids = torch.cat(all_camids)
    print(f"  Features extracted: {feats.shape}  "
          f"query={num_query}  gallery={feats.shape[0]-num_query}")

    # Split and normalise each component independently.
    d0, d1 = embed_dim, embed_dim + backbone_dim
    raw_refined = feats[:, :d0]
    raw_imgfeat = feats[:, d0:d1]
    raw_proj    = feats[:, d1:]
    norm_refined = raw_refined.norm(p=2, dim=1).mean().item()
    norm_imgfeat = raw_imgfeat.norm(p=2, dim=1).mean().item()
    norm_proj    = raw_proj.norm(p=2, dim=1).mean().item()
    norm_concat  = feats.norm(p=2, dim=1).mean().item()
    print(f"  mean L2 norms - refined: {norm_refined:.3f}  "
          f"img_feat: {norm_imgfeat:.3f}  proj: {norm_proj:.3f}  concat: {norm_concat:.3f}")

    f_refined = F.normalize(raw_refined, p=2, dim=1)
    f_imgfeat = F.normalize(raw_imgfeat, p=2, dim=1)
    f_proj    = F.normalize(raw_proj,    p=2, dim=1)
    f_concat  = F.normalize(feats,       p=2, dim=1)

    # NFC after normalisation.
    if raw_cfg.get("use_nfc", False):
        k1, k2 = raw_cfg.get("nfc_k1", 2), raw_cfg.get("nfc_k2", 2)
        print(f"  [NFC] k1={k1}  k2={k2}  N={feats.shape[0]}  applying to 4 feature sets ...", flush=True)
        f_refined = apply_nfc_split(f_refined, num_query, k1=k1, k2=k2)
        print(f"  [NFC] refined done", flush=True)
        f_imgfeat = apply_nfc_split(f_imgfeat, num_query, k1=k1, k2=k2)
        print(f"  [NFC] img_feature done", flush=True)
        f_proj    = apply_nfc_split(f_proj,    num_query, k1=k1, k2=k2)
        print(f"  [NFC] proj done", flush=True)
        f_concat  = apply_nfc_split(f_concat,  num_query, k1=k1, k2=k2)
        print(f"  [NFC] concat done", flush=True)

    # Scaling by sqrt(w) makes euclidean distance a weighted cosine sim ranking
    f_fusion  = torch.cat([w_r**0.5 * f_refined,
                            w_i**0.5 * f_imgfeat,
                            w_p**0.5 * f_proj], dim=1)  # no further norm

    def _run(f, label, norm="no"):
        e = R1_mAP_eval(num_query, max_rank=50, feat_norm=norm, reranking=False)
        e.reset()
        e.update((f, pids, camids))
        cmc, mAP, *_ = e.compute()
        r1, mp = float(cmc[0]) * 100, float(mAP) * 100
        print(f"  {label:<32}  mAP: {mp:.1f}%  Rank-1: {r1:.1f}%")
        return r1, mp

    # Orthogonal projection fusion: take refined as base, add the part of img_feature
    # that refined does not already capture (img_feature's orthogonal residual).
    # Since img_feature is the stronger feature its orthogonal component carries
    # more useful identity signal than the residual of the noisier refined feature.
    #   r_perp = f_imgfeat - (f_imgfeat A f_refined) * f_refined   [unit-norm inputs]
    #   f_orth = normalize(f_refined + I2 * r_perp)
    beta = raw_cfg.get("fusion_beta", 1.0)
    proj_coeff = (f_imgfeat * f_refined).sum(dim=1, keepdim=True)  # [N, 1] cosine sim (symmetric)
    r_perp     = f_imgfeat - proj_coeff * f_refined                # [N, D] part of img_feat aS refined
    f_orth     = F.normalize(f_refined + beta * r_perp, p=2, dim=1)

    _run(f_refined, f"refined     ({embed_dim}d)")
    _run(f_imgfeat, f"img_feature ({backbone_dim}d)")
    _run(f_proj,    f"img_proj    ({feats.shape[1]-d1}d)")
    _run(f_concat,  f"concat      ({feats.shape[1]}d)")
    _run(f_orth,    f"orth-fusion I2={beta:.1f}")
    r1, mAP = _run(f_fusion, f"fusion      w=({w_r:.1f},{w_i:.1f},{w_p:.1f})")

    if has_part_eval:
        part_feats = torch.cat(all_part_feats)            # [N, K_id * embed_dim], per-anchor L2-normed
        K_id_val   = vis_model._cached_K

        # Option A a global cosine on top-K anchors.
        f_part_cos = F.normalize(part_feats, p=2, dim=1)
        _run(f_part_cos, f"part-cos    (top-{part_k}/{K_id_val}, {part_feats.shape[1]}d)",
             norm="no")

        # Option B a score aggregation on top-K anchors (Icos, no global norm).
        _run(part_feats, f"part-score  (top-{part_k}/{K_id_val}, Icos)", norm="no")

        # Option C a all anchors (no top-K mask), global cosine.
        if all_part_feats_all and part_k < K_id_val:
            part_feats_all = torch.cat(all_part_feats_all)
            f_part_all_cos = F.normalize(part_feats_all, p=2, dim=1)
            _run(f_part_all_cos,
                 f"part-cos    (all-{K_id_val}, {part_feats_all.shape[1]}d)", norm="no")

    return r1, mAP


@torch.no_grad()
def _validate_tta_multi(vis_model, val_loader, num_query, device,
                        rerank: bool = False) -> Tuple[float, float]:
    """Option A: 4-view TTA on queries (orig + flip + pseudo-IR + pseudo-RGB).
    Image-image matching using the refined feat embedding."""
    vis_model.eval()
    eval1 = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes", reranking=False)
    eval1.reset()
    if rerank:
        eval2 = R1_mAP_eval(num_query, max_rank=50, feat_norm="yes", reranking=True)
        eval2.reset()

    processed = 0
    for imgs, pids, camids, camids_batch, viewids, _ in val_loader:
        imgs = imgs.to(device)
        cam_ids_d = camids_batch.to(device)
        if processed < num_query:
            f0 = vis_model.encode_eval(imgs, cam_ids=cam_ids_d)
            f1 = vis_model.encode_eval(imgs.flip(dims=[3]), cam_ids=cam_ids_d)
            pseudo_ir  = imgs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            pseudo_rgb = imgs[:, 0:1].repeat(1, 3, 1, 1)
            f2 = vis_model.encode_eval(pseudo_ir, cam_ids=cam_ids_d)
            f3 = vis_model.encode_eval(pseudo_rgb, cam_ids=cam_ids_d)
            feat = (f0 + f1 + f2 + f3) * 0.25
        else:
            feat = vis_model.encode_eval(imgs, cam_ids=cam_ids_d)
        processed += imgs.shape[0]
        feat = F.normalize(feat, p=2, dim=1)
        eval1.update((feat.cpu(), pids, camids))
        if rerank:
            eval2.update((feat.cpu(), pids, camids))

    cmc, mAP, *_ = eval1.compute()
    if rerank:
        cmc2, mAP2, *_ = eval2.compute()
        print(f"  [rerank] mAP: {float(mAP2)*100:.1f}%  Rank-1: {float(cmc2[0])*100:.1f}%", end="")
    return float(cmc[0]) * 100, float(mAP) * 100


def _validate_ttpt(vis_model, val_loader, num_query, device,
                   num_classes: int, raw_cfg: Dict) -> Tuple[float, float]:
    """Option B: per-query TTPT that jointly optimises prompt_learner.ctx_generic (CLIP entropy)
    and text_anchor.ctx_anchors (attention entropy over text anchors).
    Final matching uses full encode_eval features (2048-dim) for both query and gallery."""
    ttpt_lr              = raw_cfg.get("ttpt_lr",              1e-3)
    ttpt_anchor_lr       = raw_cfg.get("ttpt_anchor_lr",       ttpt_lr * 0.1)
    ttpt_steps           = raw_cfg.get("ttpt_steps",           5)
    ttpt_temp            = raw_cfg.get("ttpt_temp",            0.07)
    ttpt_cls_sample      = raw_cfg.get("ttpt_num_classes_sample", num_classes)
    ttpt_batch           = raw_cfg.get("ttpt_batch",           None)
    tta                  = raw_cfg.get("eval_tta",             False)
    s2_batch             = raw_cfg.get("stage1_batch",         64)

    try:
        ctx_param    = vis_model.clipreid_model.prompt_learner.ctx_generic
        anchor_param = vis_model.text_anchor.ctx_anchors
    except AttributeError:
        print("[TTPT] required params not accessible - falling back to standard eval")
        return _validate_stage4(vis_model, val_loader, num_query, device, tta=tta, raw_cfg=raw_cfg)

    vis_model.eval()
    ctx_orig    = ctx_param.data.clone()
    anchor_orig = anchor_param.data.clone()
    for p in vis_model.parameters():
        p.requires_grad_(False)

    K_anchors = anchor_param.shape[0]   # number of text anchors

    def _text_feats_normalized():
        if ttpt_cls_sample < num_classes:
            labels = torch.randperm(num_classes, device=device)[:ttpt_cls_sample]
        else:
            labels = torch.arange(num_classes, device=device)
        parts = []
        for i in range(0, labels.shape[0], s2_batch):
            l = labels[i:i + s2_batch]
            parts.append(vis_model.clipreid_model(label=l, get_text=True))
        return F.normalize(torch.cat(parts).float(), dim=1)  # [C', 512]

    query_feats,   query_pids,   query_camids   = [], [], []
    gallery_feats, gallery_pids, gallery_camids = [], [], []
    processed = 0

    for imgs, pids, camids, camids_batch, viewids, _ in val_loader:
        imgs = imgs.to(device)
        chunk = ttpt_batch if (ttpt_batch and processed < num_query) else imgs.shape[0]

        if processed < num_query:
            for c_start in range(0, imgs.shape[0], chunk):
                c_imgs   = imgs[c_start:c_start + chunk]
                c_pids   = pids[c_start:c_start + chunk]
                c_camids = camids[c_start:c_start + chunk]

                # Extract fixed backbone features once (no grad needed)
                with torch.no_grad():
                    feat_last, img_feat_seq, img_proj_seq = vis_model.backbone(c_imgs, token_layer=vis_model.token_layer)
                    tokens      = feat_last[:, 1:].float()       # [B, N, 768] patch tokens
                    img_feature = img_feat_seq[:, 0].float()     # [B, 768] x12 CLS
                    proj        = img_proj_seq[:, 0].float()     # [B, 512] CLIP proj
                    if tta:
                        fl, ifs, ips = vis_model.backbone(c_imgs.flip(dims=[3]), token_layer=vis_model.token_layer)
                        tokens      = (tokens      + fl[:, 1:].float())  * 0.5
                        img_feature = (img_feature + ifs[:, 0].float())  * 0.5
                        proj        = (proj        + ips[:, 0].float())  * 0.5
                proj_q = F.normalize(proj, dim=1).detach()

                # Joint TTPT optimisation
                ctx_param.data.copy_(ctx_orig)
                anchor_param.data.copy_(anchor_orig)
                ctx_param.requires_grad_(True)
                anchor_param.requires_grad_(True)
                opt = torch.optim.AdamW(
                    [{"params": [ctx_param],    "lr": ttpt_lr},
                     {"params": [anchor_param], "lr": ttpt_anchor_lr}],
                    weight_decay=0,
                )
                for _ in range(ttpt_steps):
                    # Loss 1: CLIP entropy (drives ctx_param)
                    tf    = _text_feats_normalized()
                    sim   = (proj_q @ tf.t()) / ttpt_temp
                    probs = F.softmax(sim, dim=-1)
                    loss_clip = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

                    # Loss 2: attention entropy over text anchors (drives anchor_param)
                    B_t         = tokens.shape[0]
                    text_anch   = vis_model.text_anchor()                               # [K, 768]
                    text_anch_b = text_anch.unsqueeze(0).expand(B_t, -1, -1)           # [B, K, 768]
                    if isinstance(vis_model.domain_anchor, CLIPGroundedDomainAnchorGenerator):
                        domain_anch = vis_model.domain_anchor(proj, text_anch_b)       # [B, M, 768]
                    else:
                        domain_anch = vis_model.domain_anchor(tokens)                  # [B, M, 768]
                    anchors_all = torch.cat([text_anch_b, domain_anch], dim=1)         # [B, K+M, 768]
                    _, attn_w   = vis_model.refine(tokens, anchors_all)                 # [B, N, K+M]
                    attn_text   = attn_w[:, :, :K_anchors].mean(dim=1)                 # [B, K]
                    attn_probs  = F.softmax(attn_text, dim=-1)
                    loss_anchor = -(attn_probs * torch.log(attn_probs + 1e-9)).sum(dim=-1).mean()

                    loss = loss_clip + loss_anchor
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                ctx_param.requires_grad_(False)
                anchor_param.requires_grad_(False)

                # Query feature: encode_eval with adapted anchors
                with torch.no_grad():
                    refined_feat = vis_model.encode(tokens, clip_proj=proj)             # [B, 768]
                    q_feat = F.normalize(
                        torch.cat([refined_feat, img_feature, proj], dim=1), dim=1)    # [B, 2048]

                # Restore
                ctx_param.data.copy_(ctx_orig)
                anchor_param.data.copy_(anchor_orig)
                query_feats.append(q_feat.cpu())
                query_pids.append(torch.as_tensor(c_pids))
                query_camids.append(torch.as_tensor(c_camids))

        else:
            # Gallery: standard encode_eval with original anchors
            with torch.no_grad():
                feat = F.normalize(vis_model.encode_eval(imgs, cam_ids=camids_batch.to(device)), dim=1)
            gallery_feats.append(feat.cpu())
            gallery_pids.append(torch.as_tensor(pids))
            gallery_camids.append(torch.as_tensor(camids))

        processed += imgs.shape[0]

    ctx_param.data.copy_(ctx_orig)
    anchor_param.data.copy_(anchor_orig)

    qf = torch.cat(query_feats)
    gf = torch.cat(gallery_feats)
    qp = torch.cat(query_pids)
    gp = torch.cat(gallery_pids)
    qc = torch.cat(query_camids)
    gc = torch.cat(gallery_camids)

    eval1 = R1_mAP_eval(len(qf), max_rank=50, feat_norm="no", reranking=False)
    eval1.reset()
    eval1.update((qf, qp, qc))
    eval1.update((gf, gp, gc))
    cmc, mAP, *_ = eval1.compute()
    return float(cmc[0]) * 100, float(mAP) * 100


def run_stage4(vis_model, stage2_loader, val_loader, num_query,
               num_classes: int, raw_cfg: Dict, device, ckpt_dir: str):
    epochs      = raw_cfg.get("s4_epochs",          40)
    bb_lr       = raw_cfg.get("s4_backbone_lr",     5e-6)
    head_lr     = raw_cfg.get("s4_head_lr",         5e-5)
    warmup_ep   = raw_cfg.get("s4_warmup",          5)
    save_period = raw_cfg.get("stage4_save_period", 10)

    print(f"\n[Stage 4] backbone-only fine-tune  {epochs} epochs  bb_lr={bb_lr:.1e}")

    # Unfreeze backbone only a freeze all visual modules
    for p in vis_model.parameters():
        p.requires_grad_(False)
    for p in vis_model.backbone.parameters():
        p.requires_grad_(True)

    backbone_params = list(vis_model.backbone.parameters())

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": bb_lr, "name": "backbone"}],
        weight_decay=raw_cfg["weight_decay"],
    )

    s4_min_lr_ratio = raw_cfg.get("s4_min_lr_ratio", 0.01)

    def _lr_lambda(ep):
        if ep < warmup_ep:
            return 0.1 + 0.9 * ep / max(warmup_ep, 1)
        t = (ep - warmup_ep) / max(epochs - warmup_ep, 1)
        return s4_min_lr_ratio + (1.0 - s4_min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler    = torch.amp.GradScaler("cuda")
    xent      = CrossEntropyLabelSmooth(num_classes)
    triplet   = TripletLoss(raw_cfg["margin"])
    lambda_orth = raw_cfg.get("lambda_orth", 0.0)
    best_mAP  = 0.0

    for epoch in range(1, epochs + 1):
        vis_model.train()
        total_loss = total_rxent = total_rtri = total_orth = 0.0
        t0 = time.time()

        for imgs, pids, camids, viewids in stage2_loader:
            imgs   = imgs.to(device)
            target = pids.to(device)
            cam_ids_d = camids.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logit, feat, img_feature, _ = vis_model.forward_img(imgs, cam_ids=cam_ids_d)
                l_rxent = xent(logit, target)
                l_rtri  = raw_cfg["lambda_triplet"] * triplet(feat, target)[0]
                loss    = l_rxent + l_rtri
                l_orth = 0.0
                if lambda_orth > 0:
                    l_orth = _cross_decorr_loss(feat, img_feature, lambda_orth)
                    loss   = loss + l_orth

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss  += loss.item()
            total_rxent += float(l_rxent)
            total_rtri  += float(l_rtri)
            total_orth  += float(l_orth)

        scheduler.step(epoch)
        elapsed = time.time() - t0
        n   = max(len(stage2_loader), 1)
        avg = total_loss / n
        print(f"  [S4 {epoch:3d}/{epochs}]  loss: {avg:.4f}  "
              f"rx={total_rxent/n:.3f} rt={total_rtri/n:.3f}"
              + (f" ro={total_orth/n:.3f}" if lambda_orth > 0 else "")
              + f"  ({elapsed:.0f}s)", end="")

        if epoch % save_period == 0 or epoch == epochs:
            is_final = (epoch == epochs)
            rerank   = is_final and raw_cfg.get("eval_rerank", False)
            tta      = is_final and raw_cfg.get("eval_tta",    False)
            r1, mAP = _validate_fusion(vis_model, val_loader, num_query, device, raw_cfg)
            print(f"  mAP: {mAP:.1f}%  Rank-1: {r1:.1f}%", end="")
            torch.save(vis_model.state_dict(),
                       os.path.join(ckpt_dir, f"model_stage4_ep{epoch:03d}.pth"))
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(vis_model.state_dict(),
                           os.path.join(ckpt_dir, "model_stage4_best.pth"))
                print("  <- best", end="")
        print()

    print(f"[Stage 4] done.  Best mAP: {best_mAP:.1f}%")
    return best_mAP


def run_stage4b(vis_model, stage2_loader, val_loader, num_query,
                num_classes: int, raw_cfg: Dict, device, ckpt_dir: str):
    """Stage 4b: fine-tune backbone only, visual modules frozen.

    Inverse of Stage 3 a lets the CLIP backbone adapt to the domain while
    keeping the trained visual module weights fixed to avoid overfitting.
    Loads from Stage 3 best checkpoint (caller's responsibility).
    """
    epochs      = raw_cfg.get("s4b_epochs",          40)
    bb_lr       = raw_cfg.get("s4b_backbone_lr",     5e-6)
    warmup_ep   = raw_cfg.get("s4b_warmup",          5)
    save_period = raw_cfg.get("stage4b_save_period", 10)

    print(f"\n[Stage 4b] backbone-only fine-tune  {epochs} epochs  bb_lr={bb_lr:.1e}")

    # Freeze all visual module parameters
    for p in vis_model.text_anchor.parameters():
        p.requires_grad_(False)
    for p in vis_model.domain_anchor.parameters():
        p.requires_grad_(False)
    for p in vis_model.refine.parameters():
        p.requires_grad_(False)
    for p in vis_model.bn.parameters():
        p.requires_grad_(False)
    for p in vis_model.classifiers.parameters():
        p.requires_grad_(False)
    if not isinstance(vis_model.proj, nn.Identity):
        for p in vis_model.proj.parameters():
            p.requires_grad_(False)
    for p in vis_model.text_proj.parameters():
        p.requires_grad_(False)

    # Unfreeze backbone
    for p in vis_model.backbone.parameters():
        p.requires_grad_(True)
    # Text encoder stays frozen
    if hasattr(vis_model.text_anchor, "text_encoder"):
        for p in vis_model.text_anchor.text_encoder.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        vis_model.backbone.parameters(),
        lr=bb_lr,
        weight_decay=raw_cfg["weight_decay"],
    )

    s4b_min_lr_ratio = raw_cfg.get("s4b_min_lr_ratio", raw_cfg.get("s4_min_lr_ratio", 0.1))

    def _lr_lambda(ep):
        if ep < warmup_ep:
            return 0.1 + 0.9 * ep / max(warmup_ep, 1)
        t = (ep - warmup_ep) / max(epochs - warmup_ep, 1)
        return s4b_min_lr_ratio + (1.0 - s4b_min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler    = torch.amp.GradScaler("cuda")
    xent      = CrossEntropyLabelSmooth(num_classes)
    triplet   = TripletLoss(raw_cfg["margin"])
    best_mAP  = 0.0

    # Pre-compute text features once a text encoder is frozen throughout
    s2_batch = raw_cfg.get("stage1_batch", 64)
    text_feats_all = []
    vis_model.clipreid_model.eval()
    with torch.no_grad():
        for i in range(0, num_classes, s2_batch):
            l = torch.arange(i, min(i + s2_batch, num_classes), device=device)
            with torch.amp.autocast("cuda"):
                tf = vis_model.clipreid_model(label=l, get_text=True)
            text_feats_all.append(tf.cpu())
    text_feats_all = torch.cat(text_feats_all).float().to(device)

    for epoch in range(1, epochs + 1):
        vis_model.train()
        # Keep visual modules in eval mode (weights frozen, no gradient)
        # but leave vis_model.bn in train mode so running stats update
        # as the backbone output distribution shifts during fine-tuning.
        vis_model.text_anchor.eval()
        vis_model.domain_anchor.eval()
        vis_model.refine.eval()
        vis_model.classifiers.eval()

        total_loss, t0 = 0.0, time.time()

        for imgs, pids, camids, viewids in stage2_loader:
            imgs   = imgs.to(device)
            target = pids.to(device)
            cam_ids_d = camids.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logit, feat, _, img_feat_proj = vis_model.forward_img(imgs, cam_ids=cam_ids_d)

                i2t_loss = xent(img_feat_proj @ text_feats_all.t(), target)
                loss = (xent(logit, target) +
                        raw_cfg["lambda_triplet"] * triplet(feat, target)[0] +
                        raw_cfg["lambda_i2t"]    * i2t_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step(epoch)
        elapsed = time.time() - t0
        avg = total_loss / max(len(stage2_loader), 1)
        print(f"  [S4b {epoch:3d}/{epochs}]  loss: {avg:.4f}  ({elapsed:.0f}s)", end="")

        if epoch % save_period == 0 or epoch == epochs:
            is_final = (epoch == epochs)
            rerank   = is_final and raw_cfg.get("eval_rerank", False)
            tta      = is_final and raw_cfg.get("eval_tta",    False)
            r1, mAP = _validate_fusion(vis_model, val_loader, num_query, device, raw_cfg)
            print(f"  mAP: {mAP:.1f}%  Rank-1: {r1:.1f}%", end="")
            torch.save(vis_model.state_dict(),
                       os.path.join(ckpt_dir, f"model_stage4b_ep{epoch:03d}.pth"))
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(vis_model.state_dict(),
                           os.path.join(ckpt_dir, "model_stage4b_best.pth"))
                print("  <- best", end="")
        print()

    print(f"[Stage 4b] done.  Best mAP: {best_mAP:.1f}%")
    return best_mAP


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def _load_checkpoint(model, path: str, device, label: str = ""):
    """Load checkpoint with strict=False; print any missing/unexpected keys."""
    sd = torch.load(path, map_location=device)
    result = model.load_state_dict(sd, strict=False)
    tag = f"[{label}] " if label else ""
    if result.missing_keys:
        print(f"  {tag}missing keys (will use random init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  {tag}unexpected keys (ignored): {result.unexpected_keys}")
    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=None,
                        help="Path to JSON config file (overrides built-in CFG defaults)")
    parser.add_argument("--dataset_root",  default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--start_stage",   type=int, default=1,
                        help="1=scratch, 2=skip S1, 3=visual module only, 4=fine-tune only (S4 or S4b per config)")
    parser.add_argument("--start_s3b",    action="store_true",
                        help="Skip Stage 3a and start from Stage 3b, loading model_stage3a_best.pth")
    parser.add_argument("--eval",          default=None,
                        help="Eval-only mode: path to a Stage 4 checkpoint (vis_model state_dict). "
                             "Runs evaluation with current config flags (eval_tta, eval_rerank).")
    parser.add_argument("--eval_mode",     default=None,
                        choices=["standard", "tta_multi", "ttpt", "fusion"],
                        help="Inference style: standard (default), tta_multi (4-view TTA, Option A), "
                             "ttpt (test-time prompt tuning, Option B)")
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--stage2_lr",     type=float, default=None)
    parser.add_argument("--dataset",       default=None,
                        help="dataset name: market1501 | dukemtmc | occ_duke | occ_market | occ_reid | mmmp")
    parser.add_argument("--exp_setting",   default=None,
                        help="MMMP experiment setting, e.g. exp_cctv_ir_cctv_rgb")
    args = parser.parse_args()

    cfg_d = dict(CFG)
    file_cfg = {}
    if args.config:
        # utf-8-sig handles UTF-8 files with/without BOM (common on Windows).
        with open(args.config, encoding="utf-8-sig") as f:
            file_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        if file_cfg.get("use_v1_features", False):
            cfg_d.update(V1_FEATURE_OVERRIDES)
        cfg_d.update(file_cfg)
    if args.dataset_root:   cfg_d["dataset_root"]   = args.dataset_root
    if args.checkpoint_dir: cfg_d["checkpoint_dir"] = args.checkpoint_dir
    if args.stage1_epochs:  cfg_d["stage1_epochs"]  = args.stage1_epochs
    if args.stage2_epochs:  cfg_d["stage2_epochs"]  = args.stage2_epochs
    if args.stage2_lr:      cfg_d["stage2_lr"]      = args.stage2_lr
    if args.dataset:        cfg_d["dataset"]        = args.dataset
    if args.exp_setting:    cfg_d["exp_setting"]    = args.exp_setting

    os.makedirs(cfg_d["checkpoint_dir"], exist_ok=True)

    # Mirror all stdout output to a log file in the checkpoint directory.
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data)
        def flush(self):
            for s in self.streams: s.flush()
        def isatty(self): return False

    _log_path = os.path.join(cfg_d["checkpoint_dir"], "training.log")
    _log_file = open(_log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    print(f"\n{'='*70}")
    print(f"Run started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config:      {args.config or '(built-in defaults)'}")
    print(f"Log file:    {_log_path}")
    print(f"{'='*70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cfg = _Cfg(cfg_d)

    # ---- Data ----
    stage1_loader, stage2_loader, val_loader, num_query, num_classes, cam_num = make_loaders(cfg_d, cfg)
    print(f"Dataset: {num_classes} identities  num_query={num_query}  cameras={cam_num}")

    # ---- Model ----
    sie_cam = cfg_d.get("use_sie_camera", False)
    if sie_cam:
        print(f"SIE camera enabled: {cam_num} cameras  coe={cfg_d.get('sie_coe', 3.0)}")
    model = build_transformer(num_classes, camera_num=cam_num if sie_cam else 0, view_num=0, cfg=cfg).to(device)

    # ---- Eval-only mode ----
    if args.eval:
        s2_ckpt_eval = os.path.join(cfg_d["checkpoint_dir"], "model_stage2_final.pth")
        if not os.path.exists(s2_ckpt_eval):
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {s2_ckpt_eval}")
        _load_checkpoint(model, s2_ckpt_eval, device, "S2->eval")
        vis_model = CLIPReIDWithVisual(model.image_encoder, model, num_classes, cfg_d, camera_num=cam_num).to(device)
        _load_checkpoint(vis_model, args.eval, device, "S3->eval")
        print(f"[Eval] loaded checkpoint: {args.eval}")
        rerank    = cfg_d.get("eval_rerank", False)
        eval_mode = args.eval_mode or cfg_d.get("eval_mode", "fusion")
        if eval_mode == "tta_multi":
            print("[Eval] Mode: 4-view TTA (Option A)")
            r1, mAP = _validate_tta_multi(vis_model, val_loader, num_query, device, rerank=rerank)
        elif eval_mode == "ttpt":
            print("[Eval] Mode: TTPT (Option B)  "
                  f"lr={cfg_d.get('ttpt_lr',1e-3)}  steps={cfg_d.get('ttpt_steps',5)}")
            r1, mAP = _validate_ttpt(vis_model, val_loader, num_query, device, num_classes, cfg_d)
        elif eval_mode == "fusion":
            print("[Eval] Mode: score-level fusion")
            r1, mAP = _validate_fusion(vis_model, val_loader, num_query, device, cfg_d)
        else:
            tta = cfg_d.get("eval_tta", False)
            print(f"[Eval] Mode: standard  tta={tta}  rerank={rerank}")
            r1, mAP = _validate_stage4(vis_model, val_loader, num_query, device,
                                       rerank=rerank, tta=tta, raw_cfg=cfg_d)
        print(f"mAP: {mAP:.1f}%  Rank-1: {r1:.1f}%")
        return

    s1_ckpt = os.path.join(cfg_d["checkpoint_dir"], "model_stage1.pth")
    s2_ckpt = os.path.join(cfg_d["checkpoint_dir"], "model_stage2_final.pth")
    s3_ckpt = os.path.join(cfg_d["checkpoint_dir"], "model_stage3_best.pth")

    if args.start_stage <= 1:
        run_stage1(model, stage1_loader, val_loader, num_query, cfg_d, device, cfg_d["checkpoint_dir"])

    if args.start_stage <= 2:
        if args.start_stage == 2:
            if not os.path.exists(s1_ckpt):
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1_ckpt}")
            _load_checkpoint(model, s1_ckpt, device, "S1")
            print(f"Loaded Stage 1 checkpoint from {s1_ckpt}")
        run_stage2(model, stage2_loader, val_loader, num_query, num_classes, cfg_d, device,
                   cfg_d["checkpoint_dir"])

    vis_model = None
    if args.start_stage <= 3:
        if args.start_stage == 3 or args.start_s3b:
            if not os.path.exists(s2_ckpt):
                raise FileNotFoundError(f"Stage 2 checkpoint not found: {s2_ckpt}")
            _load_checkpoint(model, s2_ckpt, device, "S2")
            print(f"Loaded Stage 2 checkpoint from {s2_ckpt}")
        run_stage3(model, stage2_loader, val_loader, num_query, num_classes,
                   cfg_d, device, cfg_d["checkpoint_dir"],
                   skip_s3a=args.start_s3b, cam_num=cam_num)

    if args.start_stage <= 4:
        if not cfg_d.get("enable_stage4", False):
            print("[Stage 4] disabled by config (enable_stage4=false); skipping Stage 4.")
        else:
            # Build vis_model and load Stage 3 best weights
            if not os.path.exists(s2_ckpt):
                raise FileNotFoundError(f"Stage 2 checkpoint not found: {s2_ckpt}")
            if args.start_stage == 4:
                _load_checkpoint(model, s2_ckpt, device, "S2")
            vis_model = CLIPReIDWithVisual(model.image_encoder, model, num_classes, cfg_d, camera_num=cam_num).to(device)
            if not os.path.exists(s3_ckpt):
                raise FileNotFoundError(f"Stage 3 checkpoint not found: {s3_ckpt}")
            _load_checkpoint(vis_model, s3_ckpt, device, "S3")
            print(f"Loaded Stage 3 checkpoint from {s3_ckpt}")
            if cfg_d.get("use_s4b", False):
                run_stage4b(vis_model, stage2_loader, val_loader, num_query, num_classes,
                            cfg_d, device, cfg_d["checkpoint_dir"])
            else:
                run_stage4(vis_model, stage2_loader, val_loader, num_query, num_classes,
                           cfg_d, device, cfg_d["checkpoint_dir"])


if __name__ == "__main__":
    main()




