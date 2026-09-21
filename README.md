# RARNet

FSC147 inference for **Reliability-Aware Readout Network (RARNet)**. The complete model always includes **Exemplar Reliability Aggregation (ERA)** and **Ordinal Count Allocation (OCA)**.

## Setup

Use a separate Python 3.8 environment. Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

The pinned PyTorch build uses CUDA 11.1; an NVIDIA driver compatible with that runtime is needed for GPU execution. CPU inference is also supported, but is slower.

## Data and checkpoint

Place the prepared FSC147 files in this structure:

```text
data/FSC147/
  images_384_VarV2/
  annotation_FSC147_384.json
  Train_Test_Val_FSC_147.json
weights/
  rarnet.pth
```

Use the official FSC147 prepared images and annotations from [FSC147 / FamNet](https://github.com/cvlab-stonybrook/LearningToCountEverything). Density-map ground truth is not required. A compatible complete RARNet checkpoint must be supplied separately; weights and datasets are not bundled in this repository.

The loader accepts a tensor state dictionary or a trusted checkpoint containing a `model` state dictionary. Historical parameter names are translated by `models/checkpoint_compat.py`; model parameters are then loaded strictly. Retained inactive parameter slots support existing checkpoint files.

## Run

Edit these three paths in `run_fsc147.py`:

```python
DATA_ROOT = ROOT / 'data' / 'FSC147'
CHECKPOINT = ROOT / 'weights' / 'rarnet.pth'
OUTPUT_ROOT = ROOT / 'outputs' / 'fsc147'
```

Then run:

```bash
python run_fsc147.py
```

The script evaluates validation (1,286 images) and test (1,190 images) using the same supplied checkpoint. It writes per-image predictions and MAE/RMSE summaries into a new timestamped output directory. No parameter selection occurs during evaluation.

## Fixed inference protocol

This release uses the historical FSC147 protocol: height-384 prepared images, three 64×64 exemplar crops, 384×384 horizontal windows with stride 128, and sequential averaging of overlapping predictions. Density values are divided by 60 before summation. Counts are normalized by the mean count inside the three exemplar boxes when that mean exceeds 1.8.

For an exemplar smaller than 10 pixels in both dimensions, the historical 3×3 crop-and-resize procedure is applied. Its summed count is used unless it exceeds nine times the original estimate. There is no multiscale ensemble. Evaluation uses FP32 with TF32 disabled. ERA temperature is fixed to 0.25 and its concentration/consensus weights to 1.0.

## Layout

```text
run_fsc147.py              # Path configuration and entry point
inference/                # FSC147 data loading, windows and evaluation
models/RARNet.py          # Complete model forward pass
models/Block/             # Transformer blocks
models/checkpoint_compat.py
util/pos_embed.py
```

This release contains inference code only. The model implementation builds on CACViT; the upstream copyright and license are retained in `LICENSE`.
