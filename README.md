# Structured Anchor-Guided Aggregation for ReID

Official training/evaluation code for the paper.

Paper: [From Global to Local: Rethinking CLIP Feature Aggregation for Person Re-Identification](https://arxiv.org/abs/2604.22190)

## What is included

This open-source package includes the standalone training/evaluation pipeline:

- `train.py` is the main entrypoint

## Environment setup

Tested with Python 3.8+, PyTorch 2.4, CUDA 12.

### 1. Create and activate an environment

Option A (Conda, recommended):

```bash
conda create -n saga-reid python=3.10 -y
conda activate saga-reid
```

Option B (venv):

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch

Install the PyTorch build that matches your system and CUDA version.

CUDA 12.1 example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
python -c "import timm, ftfy, regex, PIL, scipy, sklearn, numpy; print('dependencies_ok')"
```

## Supported datasets

- `market1501`
- `dukemtmc`
- `occ_duke`
- `occ_market`
- `occ_reid`
- `mmmp`

## Training

```bash
python train.py --config configs/market1501.json
```

### Switch to old v1-style features (config only)

Set this in your config file:

```json
"use_v1_features": true
```

## Resume

```bash
# resume from Stage-2 checkpoint
python train.py --config configs/market1501.json --start_stage 3

# start Stage 3b directly
python train.py --config configs/market1501.json --start_stage 3 --start_s3b
```

Stage 4 training is disabled by default (`"enable_stage4": false`).

## Evaluation

```bash
# fusion (default, using Stage-3 checkpoint)
python train.py --config configs/market1501.json --eval output/structured_anchor_guided/model_stage3_best.pth
```

## Repository layout

```text
train.py
configs/
  market1501.json
  occ_market1501.json
  occ_dukemtmcreid.json
datasets/
loss/
model/
utils/
visual_modules.py
```

