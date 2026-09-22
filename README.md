# BAAR: Boundary-Aware Attention Refinement for Ultrasound Image Segmentation

PyTorch implementation of **BAAR**, a lightweight refinement module for binary
ultrasound segmentation. Multi-scale boundary attention supplies context to a
local KEEP / ADD / REMOVE predictor. Bounded logit corrections are applied
within a candidate band derived from the coarse prediction.

The workflow has two stages: train a segmentation backbone, then freeze its
parameters and batch-normalization statistics while training BAAR.
Supported backbones: **ESPNet, UNeXt-S, UNeXt and CMUNeXt**.

## 1. Install

Use Python 3.11. The paper configuration uses PyTorch 2.5.1 and CUDA 11.8.

~~~bash
python -m venv .venv
~~~

Activate the environment with `.\.venv\Scripts\Activate.ps1` in Windows
PowerShell or `source .venv/bin/activate` on Linux/macOS.

For an NVIDIA GPU:

~~~bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
~~~

For CPU execution, replace `cu118` with `cpu` in the installation command.
Run the following commands from the repository root.
`--device auto` selects CUDA when available; `--device cpu` selects CPU.
Use `python -m baar --help` to list commands.

## 2. Prepare data

Supply your dataset through `--data-root`:

~~~text
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
~~~

Each image and mask must have the same filename stem and original dimensions,
for example `sample_001.jpg` and `sample_001.png`. Supported formats are PNG,
JPEG, BMP and TIFF. Use lossless masks: zero is background; any positive value
is foreground.

Images are converted to grayscale, resized to 256 × 256 with bilinear
interpolation and normalized to [0, 1]. Masks use nearest-neighbor resizing.
Paired spatial augmentation and image-intensity augmentation apply only during
training. Prepare the splits before training; training uses `train` and `val`.

The repository contains source code. Datasets, image-level records, predictions
and trained weights are not bundled; dataset paths are supplied at runtime.

## 3. Train the baseline

Replace `/path/to/dataset` with your dataset location; quote paths with spaces.

~~~bash
python -m baar train --stage baseline --backbone unext --data-root "/path/to/dataset" --output-dir outputs/unext_baseline --seed 3407
~~~

Backbone choices: `espnet`, `unext_s`, `unext`, `cmunext`.

Defaults: up to 200 epochs, batch size 8, AdamW, learning rate 0.0001,
cosine decay to 0.000001, weight decay 0.0001 and gradient clipping at norm 1.0.
Validation-loss early stopping starts at epoch 40 with patience 12;
an improvement must exceed max(0.0001, 0.002 × |best loss|).
The selected checkpoint maximizes validation Dice, then minimizes HD95,
with the earliest epoch retained for exact ties.

Outputs: `best.pt`, `history.json` and `selection.json`.
Use a fresh output directory for each run.

## 4. Train BAAR

~~~bash
python -m baar train --stage baar --baseline outputs/unext_baseline/best.pt --config configs/baar.json --data-root "/path/to/dataset" --output-dir outputs/unext_baar --seed 3407
~~~

The backbone and image size are read from the baseline checkpoint.
BAAR uses 20 epochs, learning rate 0.001 and cosine decay to 0.00001.
Correction strength follows a half-cosine ramp over the first six epochs;
validation and inference use strength 1.0. Training uses FP32.

Selection maximizes the mean of BF1@2 and BDice-w3 subject to validation Dice
being at least the baseline Dice minus 0.005. Exact ties retain the earliest
epoch. If no epoch meets the criterion, `selection.json` records the outcome
and no selected checkpoint is written.

The BAAR checkpoint contains both backbone and refiner weights.

## 5. Evaluate and predict

~~~bash
python -m baar evaluate --checkpoint outputs/unext_baar/best.pt --data-root "/path/to/dataset" --split test --output outputs/unext_baar/test_metrics.json
python -m baar predict --checkpoint outputs/unext_baar/best.pt --input-dir "/path/to/images" --output-dir outputs/predictions
~~~

Evaluation reports image-averaged **BF1@2, BDice-w3, HD95, ASSD, Dice and IoU**.
The same commands accept a baseline checkpoint.
HD95 and ASSD use pixels on the resized image grid.

Prediction saves 0/255 PNG masks at the model's input resolution.
Files are named `prediction_000001.png`, `prediction_000002.png`, etc., in the
lexicographic order of input paths. Original image names are not copied to
outputs or logs.

## 6. Configuration and code

[configs/baar.json](configs/baar.json) provides the model and loss defaults.

| Setting | Default |
| --- | --- |
| Attention grid / dimension / heads | 16 × 16 / 32 / 4 |
| ASPP branch channels / dilation rates | 4 / 3 and 6 |
| Attention dropout / initial alpha | 0.1 / 0.1 |
| Attention gate radius / threshold | 2 / 0.5 |
| Candidate radius / hidden channels | 4 / 16 |
| Action temperature / crossing margin | 1.0 / 0.25 |
| Maximum directional proposal | 6.0 |
| Dice / GT-band Dice / BCE / distance-field weights | 0.4 / 0.3 / 0.2 / 0.1 |
| GT-band radius / distance truncation | 3 / 10 |

| Source | Purpose |
| --- | --- |
| `attention.py` | Lightweight ASPP, boundary gating, self-attention and cross-attention |
| `refinement.py` | Local cues, three-branch fusion and supported logit correction |
| `losses.py` | Four-component final-segmentation objective |
| `backbones.py` | Backbone adapters and frozen two-stage integration |
| `data.py`, `augmentation.py` | Paired data loading and training augmentation |
| `metrics.py` | Six segmentation metrics |
| `engine.py`, `cli.py` | Training, selection, evaluation and prediction |
| `_backbones/` | Required upstream sources and licenses |

All source paths are under `src/baar/`. For a custom backbone, keep it frozen
and in evaluation mode, obtain detached decoder features and coarse logits,
and apply `BAAR(feature_channels=C)`:

~~~python
from baar import BAAR

refiner = BAAR(feature_channels=16).to(image.device)
# image: [B,1,H,W]; features: [B,16,Hf,Wf]; coarse_logits: [B,1,Hl,Wl]
refined_logits = refiner(image, features.detach(), coarse_logits.detach())
~~~

All tensors and the refiner must share a device. Under the default configuration,
logits outside the candidate band remain exactly equal to the coarse logits in
training and inference.

For component ablations, set one of `hard_local_support`, `attention_context`,
`attention_self_attention_enabled` or `attention_cross_attention_enabled` to
`false` in a configuration copy. Setting a loss weight to zero removes its
contribution; the other weights stay unchanged. CLI options include `--epochs`,
`--batch-size`, `--workers` and `--ramp-epochs` (0 enables full correction from
the first epoch).

Model and loss morphology use in-grid neighbors at image borders. Evaluation
bands use zero exterior background: radius 1 for BF1/distances and radius 3
for BDice. HD95 is the maximum of the two directional 95th percentiles; ASSD
averages the two directional means. If either boundary is empty, BF1/BDice
are zero and HD95/ASSD equal the image diagonal. Region Dice and IoU are zero
when both masks are empty.

## 7. Verify

~~~bash
python -m unittest discover -s tests -v
~~~

Tests use generated tensors and geometric masks to check parameter counts,
frozen backbones, candidate-band identity, losses, checkpoint selection and
a short train/evaluate/predict workflow.

The Git file allowlist includes source, configuration, documentation and tests.
Local datasets, weights, logs and generated outputs are ignored.

## Acknowledgments

BAAR uses the ESPNet, UNeXt and CMUNeXt backbones. Source revisions, adaptations
and retained MIT licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
