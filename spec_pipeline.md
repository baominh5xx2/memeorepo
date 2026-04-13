# DAMP-ES: End-to-End Pipeline

> **Domain-Adaptive Mutual Prompting + Efficient Segmentation**  
> DAMP (CVPR 2024) → CLIP-ES (CVPR 2023) → DeepLab  
> Repos: [DAMP](https://github.com/TL-UESTC/DAMP) | [CLIP-ES](https://github.com/linyq2117/CLIP-ES) | [D-CAM](https://github.com/JingjunYi/D-CAM)

---

## 0. Environment Setup

```bash
# Python >= 3.8, CUDA >= 11.1
conda create -n damp_es python=3.9
conda activate damp_es

# Core deps
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/openai/CLIP.git
pip install timm einops pydensecrf scipy scikit-image scikit-learn tqdm wandb

# DAMP needs Dassl
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch && pip install -e . && cd ..

# Clone main repos
git clone https://github.com/TL-UESTC/DAMP.git
git clone https://github.com/linyq2117/CLIP-ES.git
git clone https://github.com/JingjunYi/D-CAM.git   # for baseline reference only
```

---

## 1. Project Structure
damp_es/
├── configs/
│ ├── stage1_damp.yaml
│ ├── stage2_cam.yaml
│ └── stage3_seg.yaml
├── datasets/
│ ├── voc12.py
│ ├── crossdomain_seg.py # Hist / BCSS / WSSS
│ └── office_home.py # optional, for DAMP reproduction
├── stage1_damp/
│ ├── model.py # DAMP mutual prompting wrapper
│ ├── train.py
│ └── extract_features.py # export adapted visual features
├── stage2_cam/
│ ├── softmax_gradcam.py
│ ├── prompt_selection.py # sharpness-based
│ ├── synonym_fusion.py
│ ├── caa.py # class-aware attention affinity
│ └── generate_pseudomasks.py # runs CRF, exports masks
├── stage3_seg/
│ ├── deeplab.py
│ ├── cgl.py # confidence-guided loss
│ └── train_seg.py
├── tools/
│ ├── eval_miou.py
│ ├── eval_crossdomain.py # mIoU + FwIoU + ACC
│ └── visualize.py
└── README.md

text

---

## 2. Dataset Setup

### 2.1 Track A — Quick Sanity (VOC2012)

```bash
# Download VOC2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar -C data/

# Download SBD augmented masks (CLIP-ES uses train_aug with 10,582 images)
# https://www.sun11.me/research/aug/benchmark.tgz
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -xf benchmark.tgz -C data/SBD/

# Final structure expected by CLIP-ES
data/VOC2012/
├── Annotations/
├── ImageSets/Segmentation/{train,val,test}.txt
├── JPEGImages/
├── SegmentationClass/
└── SegmentationClassAug/   # merged from SBD
```

### 2.2 Track B — Cross-Domain Main Benchmark

Protocol nguồn từ D-CAM (MICCAI 2025). Ba dataset:
- **Hist** = LUAD-HistoSeg
- **BCSS** = BCSS-WSSS  
- **WSSS** = WSSS4LUAD

Label space thống nhất: chỉ giữ **2 class chung** giữa 3 domain.

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | background | - |
| 1 | tumor | cancerous tissue |
| 2 | stroma | stromal tissue |
| 255 | ignore | uncertain / boundary |

```bash
data/CrossDomainSeg/
├── Hist/
│   ├── images/          # .jpg hoặc .png
│   ├── masks/           # pixel label, giá trị {0,1,2,255}
│   └── splits/
│       ├── train.txt    # dùng để generate pseudo masks + train seg
│       ├── val.txt
│       └── test.txt     # chỉ dùng để eval cuối
├── BCSS/
│   └── ...              # cấu trúc giống Hist
└── WSSS/
    └── ...
```

**Cross-domain experiments cần chạy** (theo D-CAM protocol):

| Exp ID | Source → Target | Mục đích |
|--------|-----------------|----------|
| XD-1 | Hist → BCSS | main experiment |
| XD-2 | Hist → WSSS | main experiment |
| XD-3 | BCSS → Hist | main experiment |
| XD-4 | BCSS → WSSS | supplementary |

> **Rule:** target `train` chỉ được dùng làm **unlabeled images** để generate pseudo masks.  
> target `masks` chỉ được mở ra ở bước **evaluation cuối** — không được dùng trong training.

### 2.3 Track C — DAMP Reproduction (Optional)

```bash
# Office-Home đã có download script trong DAMP repo
cd DAMP/
python tools/download_officehome.py --root data/OfficeHome/

# Hoặc Mini-DomainNet
python tools/download_minidomainnet.py --root data/MiniDomainNet/
```

Track này chỉ chạy Stage 1, không cần segmentation.

---

## 3. Stage 1 — DAMP Domain Adaptation

**Mục tiêu:** từ source images có weak labels + target images không có label, học visual embedding `v'` và text embedding `s'_k` bền với domain shift, qua mutual prompting giữa image branch và text branch.

### 3.1 Config `configs/stage1_damp.yaml`

```yaml
# === Backbone ===
backbone: "ViT-B/16"           # CLIP ViT-B/16
freeze_clip: true               # không finetune CLIP encoder gốc

# === Mutual Prompting ===
n_ctx_visual: 4                 # số visual context tokens
n_ctx_text: 4                   # số text context tokens
ctx_init: ""                    # random init
class_token_position: "end"

# === Training ===
dataset: "office_home"          # hoặc "crossdomain_seg"
source_domain: "Art"
target_domain: "Real_World"
batch_size: 32
epochs: 30
warmup_epochs: 2
lr: 2.0e-4
weight_decay: 1.0e-5
optimizer: "AdamW"

# === Loss weights ===
lambda_cls: 1.0                 # source classification loss
lambda_contrast: 0.5            # instance-level contrastive
lambda_consistency: 0.3         # target consistency / pseudo-label

# === Checkpoint ===
output_dir: "checkpoints/stage1/"
save_every: 5
```

### 3.2 Train Stage 1

```bash
# Dùng DAMP repo gốc
cd DAMP/
python train.py \
  --config ../configs/stage1_damp.yaml \
  --source Art \
  --target Real_World \
  --output ../checkpoints/stage1/art_to_real/

# Hoặc nếu bạn wrap lại
python stage1_damp/train.py --config configs/stage1_damp.yaml
```

### 3.3 Export Adapted Features

```python
# stage1_damp/extract_features.py
# Dùng sau khi Stage 1 converge
# Output: adapted visual features để Stage 2 dùng thay CLIP raw features

import torch
import clip
from stage1_damp.model import DAMPWrapper

ckpt = torch.load("checkpoints/stage1/best.pth")
model = DAMPWrapper(backbone="ViT-B/16")
model.load_state_dict(ckpt)
model.eval()

# Hook vào layer trước self-attention cuối, đúng theo CLIP-ES
# model.image_encoder.transformer.resblocks[-2]   ← feature map dùng cho CAM
```

> **Key point:** Stage 2 sẽ lấy feature từ layer hook này thay vì từ CLIP raw encoder.  
> Đây là điểm khác biệt duy nhất so với CLIP-ES nguyên bản.

---

## 4. Stage 2 — CAM Generation & Pseudo-Mask

**Mục tiêu:** dùng DAMP-adapted backbone để sinh class activation maps (CAMs) trên **target images**, refine bằng CAA, post-process bằng dense CRF, xuất pseudo masks.

### 4.1 Config `configs/stage2_cam.yaml`

```yaml
# === Model ===
clip_backbone: "ViT-B/16"
damp_ckpt: "checkpoints/stage1/best.pth"
use_damp_features: true         # false = ablation: CLIP-ES nguyên bản
feature_layer: -2               # layer trước self-attention cuối

# === Softmax-GradCAM ===
use_softmax_gradcam: true       # false = ablation: vanilla Grad-CAM
replace_cls_with_avg: true      # thay class token bằng avg các patch tokens

# === Background suppression ===
use_bg_suppression: true
bg_classes:                     # class-related background set
  - "background"
  - "tissue"
  - "slide"

# === Prompt strategy ===
prompt_template: "a clean origami of {classname}."  # default từ CLIP-ES
use_sharpness_selection: true
use_synonym_fusion: true
synonyms:
  tumor: ["tumor", "cancer", "neoplasm", "carcinoma"]
  stroma: ["stroma", "stromal tissue", "connective tissue"]

# === CAA Refinement ===
use_caa: true                   # false = ablation
caa_threshold: 0.4              # 0.4 cho VOC, 0.7 cho COCO
caa_iterations: 2

# === CRF Post-processing ===
use_crf: true
crf_iter: 10
crf_pos_w: 3
crf_bi_w: 5
crf_bi_xy_std: 80
crf_bi_rgb_std: 13

# === Confidence Filtering ===
confidence_threshold: 0.25      # pixel dưới ngưỡng → ignore (255)

# === Output ===
target_data_root: "data/CrossDomainSeg/BCSS/"
target_split: "train"
pseudo_mask_output: "pseudo_masks/stage2/hist_to_bcss/"
```

### 4.2 Softmax-GradCAM

```python
# stage2_cam/softmax_gradcam.py
import torch
import torch.nn.functional as F

def softmax_gradcam(model, image, class_idx, feature_layer):
    """
    Khác vanilla GradCAM ở chỗ:
    gradient được tính từ score SAU softmax, không phải logit.
    Điều này làm các class cạnh tranh nhau → giảm confusion.
    """
    features = {}
    grads = {}

    def fwd_hook(module, input, output):
        features['feat'] = output

    def bwd_hook(module, grad_in, grad_out):
        grads['grad'] = grad_out

    handle_fwd = feature_layer.register_forward_hook(fwd_hook)
    handle_bwd = feature_layer.register_full_backward_hook(bwd_hook)

    logits = model(image)                        # (1, num_classes)
    probs = F.softmax(logits, dim=-1)            # ← softmax step

    model.zero_grad()
    score = probs[0, class_idx]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    # Weighted combination
    weights = grads['grad'].mean(dim=, keepdim=True)  # global avg pool[1][2]
    cam = (weights * features['feat']).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam
```

### 4.3 CAA Refinement

```python
# stage2_cam/caa.py
import torch
from torch.nn.functional import normalize

def sinkhorn_normalize(A, n_iter=3):
    for _ in range(n_iter):
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        A = A / (A.sum(dim=-2, keepdim=True) + 1e-8)
    return A

def caa_refine(cam, attn_weights, threshold=0.4, n_iter=2):
    """
    cam:          (H*W,)    — flattened CAM for one class
    attn_weights: (H*W, H*W) — MHSA attention từ CLIP ViT
    """
    H_W = cam.shape

    # Symmetrize attention → doubly stochastic
    D = sinkhorn_normalize(attn_weights)
    A = (D + D.T) / 2.0                # symmetric affinity

    # Bounding box mask từ CAM seeds
    cam_2d = cam.reshape(int(H_W**0.5), -1)
    mask = (cam_2d > threshold).float()

    # Tìm bbox từ connected regions
    rows = mask.any(dim=1)
    cols = mask.any(dim=0)
    if rows.any() and cols.any():
        r_min, r_max = rows.nonzero(), rows.nonzero()[-1]
        c_min, c_max = cols.nonzero(), cols.nonzero()[-1]
        box_mask = torch.zeros_like(cam_2d)
        box_mask[r_min:r_max+1, c_min:c_max+1] = 1.0
    else:
        box_mask = torch.ones_like(cam_2d)

    box_mask_flat = box_mask.reshape(-1)  # (H*W,)

    # Iterative refinement
    refined = cam.clone()
    for _ in range(n_iter):
        masked_A = A * box_mask_flat.unsqueeze(0)
        refined = (masked_A @ refined.unsqueeze(-1)).squeeze(-1)
        refined = refined / (refined.max() + 1e-8)

    return refined.reshape(cam_2d.shape)
```

### 4.4 Dense CRF

```python
# stage2_cam/generate_pseudomasks.py  (excerpt)
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

def apply_dense_crf(image_np, cam_prob, n_classes=3,
                    crf_iter=10, pos_w=3, bi_w=5,
                    bi_xy_std=80, bi_rgb_std=13):
    """
    image_np:  (H, W, 3) uint8
    cam_prob:  (n_classes, H, W) float32, sum-to-1 per pixel
    """
    H, W = image_np.shape[:2]
    d = dcrf.DenseCRF2D(W, H, n_classes)

    unary = unary_from_softmax(cam_prob)
    d.setUnaryEnergy(unary)

    # Positional kernel
    d.addPairwiseGaussian(sxy=pos_w, compat=3)

    # Bilateral kernel
    d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std,
                           rgbim=image_np, compat=bi_w)

    Q = d.inference(crf_iter)
    return np.argmax(Q, axis=0).reshape(H, W)


def export_pseudomask(cam_per_class, image_np, confidence_threshold=0.25,
                      output_path=None):
    """
    cam_per_class: dict {class_id: cam_2d (H,W)}
    Trả về pseudo mask (H,W), uncertain → 255
    """
    import torch, os
    n_cls = len(cam_per_class) + 1   # +1 for background
    H, W = list(cam_per_class.values()).shape

    prob = np.zeros((n_cls, H, W), dtype=np.float32)
    prob = 0.2   # prior background

    for cls_id, cam in cam_per_class.items():
        prob[cls_id] = cam.numpy()

    prob = prob / (prob.sum(axis=0, keepdims=True) + 1e-8)

    pseudo = apply_dense_crf(image_np, prob)

    # Confidence filter → ignore
    confidence = prob.max(axis=0)
    pseudo[confidence < confidence_threshold] = 255

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        from PIL import Image
        Image.fromarray(pseudo.astype(np.uint8)).save(output_path)

    return pseudo
```

### 4.5 Chạy Stage 2

```bash
python stage2_cam/generate_pseudomasks.py \
  --config configs/stage2_cam.yaml \
  --source_domain Hist \
  --target_domain BCSS \
  --split train \
  --output_dir pseudo_masks/hist_to_bcss/

# Log mIoU của pseudo masks so với target ground truth (chỉ để monitor, không train với)
python tools/eval_miou.py \
  --pred_dir pseudo_masks/hist_to_bcss/ \
  --gt_dir data/CrossDomainSeg/BCSS/masks/ \
  --split train \
  --n_classes 3 \
  --ignore_label 255
```

---

## 5. Stage 3 — Target Segmentation Retraining

**Mục tiêu:** dùng pseudo masks từ Stage 2 làm supervision để train segmentation network trên target domain. Không cần ground-truth pixel labels của target.

### 5.1 Config `configs/stage3_seg.yaml`

```yaml
# === Model ===
backbone: "resnet101"           # hoặc "resnet38"
seg_head: "deeplabv2"           # deeplabv2 | deeplabv3
pretrained: "imagenet"

# === Data ===
target_domain: "BCSS"
target_data_root: "data/CrossDomainSeg/BCSS/"
pseudo_mask_dir: "pseudo_masks/hist_to_bcss/"
image_size: 512
n_classes: 3                    # background, tumor, stroma
ignore_label: 255

# === Training ===
batch_size: 8
epochs: 30
lr: 2.0e-4
weight_decay: 1.0e-4
optimizer: "AdamW"
lr_scheduler: "poly"
warmup_epochs: 2
augmentation:
  random_flip: true
  random_crop: true
  color_jitter: true

# === Loss ===
use_cgl: true                   # Confidence-Guided Loss
cgl_threshold: 0.25             # ignore pixel nếu confidence < threshold
cgl_lambda: 1.0

# === Checkpoint ===
output_dir: "checkpoints/stage3/hist_to_bcss/"
val_every: 5
```

### 5.2 Confidence-Guided Loss (CGL)

```python
# stage3_seg/cgl.py
import torch
import torch.nn.functional as F

def confidence_guided_loss(pred, pseudo_mask, cam_confidence,
                           threshold=0.25, ignore_label=255):
    """
    pred:           (B, C, H, W) — logits từ segmenter
    pseudo_mask:    (B, H, W)   — pseudo labels, ignore=255
    cam_confidence: (B, H, W)   — max CAM prob per pixel
    threshold:      float       — pixel dưới ngưỡng này bị ignore

    Logic:
    - Pixel đã là 255 (ignore) → skip
    - Pixel có confidence < threshold → skip
    - Pixel còn lại → cross-entropy thông thường
    """
    B, C, H, W = pred.shape

    # Mask: chỉ giữ pixel confident
    valid = (pseudo_mask != ignore_label) & (cam_confidence >= threshold)

    if valid.sum() == 0:
        return pred.sum() * 0.0   # tránh NaN

    pred_valid  = pred.permute(0,2,3,1)[valid]   # (N, C)
    label_valid = pseudo_mask[valid]              # (N,)

    loss = F.cross_entropy(pred_valid, label_valid)
    return loss
```

### 5.3 Train Stage 3

```bash
python stage3_seg/train_seg.py \
  --config configs/stage3_seg.yaml \
  --pseudo_mask_dir pseudo_masks/hist_to_bcss/ \
  --output_dir checkpoints/stage3/hist_to_bcss/

# Evaluate
python tools/eval_crossdomain.py \
  --ckpt checkpoints/stage3/hist_to_bcss/best.pth \
  --config configs/stage3_seg.yaml \
  --split test \
  --n_classes 3 \
  --output_csv results/hist_to_bcss.csv
```

---

## 6. Evaluation

### 6.1 Metrics

```python
# tools/eval_crossdomain.py  (excerpt)
import numpy as np

def compute_metrics(preds, gts, n_classes=3, ignore_label=255):
    """
    preds, gts: list of (H,W) np.ndarray
    Returns: mIoU, FwIoU, ACC — đúng theo D-CAM paper
    """
    hist = np.zeros((n_classes, n_classes), dtype=np.float64)

    for pred, gt in zip(preds, gts):
        valid = gt != ignore_label
        p = pred[valid]
        g = gt[valid]
        hist += np.bincount(n_classes * g + p,
                            minlength=n_classes**2).reshape(n_classes, n_classes)

    # IoU per class
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / (union + 1e-8)
    mIoU = iou.mean()

    # FwIoU
    freq = hist.sum(axis=1) / (hist.sum() + 1e-8)
    FwIoU = (freq * iou).sum()

    # Pixel accuracy
    ACC = intersection.sum() / (hist.sum() + 1e-8)

    return {
        "mIoU": round(float(mIoU * 100), 2),
        "FwIoU": round(float(FwIoU * 100), 2),
        "ACC": round(float(ACC * 100), 2),
        "IoU_per_class": {f"cls_{i}": round(float(iou[i]*100),2) for i in range(n_classes)}
    }
```

### 6.2 Pseudo-mask Quality

```bash
# Chạy trước khi Stage 3 để monitor chất lượng pseudo mask
python tools/eval_miou.py \
  --pred_dir pseudo_masks/hist_to_bcss/ \
  --gt_dir data/CrossDomainSeg/BCSS/masks/ \
  --split train \
  --n_classes 3 \
  --ignore_label 255 \
  --output_csv results/pseudomask_quality.csv
```

---

## 7. Baseline Commands

Chạy đúng thứ tự này để có đủ baseline cho paper.

```bash
# ── Baseline 1: CLIP-ES nguyên bản ──────────────────────────────────────
python stage2_cam/generate_pseudomasks.py \
  --config configs/stage2_cam.yaml \
  --use_damp_features false \     # ← CLIP raw, không DAMP
  --output_dir pseudo_masks/b1_clipes/

python stage3_seg/train_seg.py \
  --pseudo_mask_dir pseudo_masks/b1_clipes/ \
  --output_dir checkpoints/b1_clipes/

python tools/eval_crossdomain.py --ckpt checkpoints/b1_clipes/best.pth \
  --output_csv results/b1_clipes.csv


# ── Baseline 2: DAMP + vanilla Grad-CAM ─────────────────────────────────
python stage2_cam/generate_pseudomasks.py \
  --config configs/stage2_cam.yaml \
  --use_damp_features true \      # ← DAMP features
  --use_softmax_gradcam false \   # ← vanilla GradCAM
  --output_dir pseudo_masks/b2_damp_vanilla/

python stage3_seg/train_seg.py \
  --pseudo_mask_dir pseudo_masks/b2_damp_vanilla/ \
  --output_dir checkpoints/b2_damp_vanilla/

python tools/eval_crossdomain.py --ckpt checkpoints/b2_damp_vanilla/best.pth \
  --output_csv results/b2_damp_vanilla.csv


# ── Baseline 3: DAMP + softmax-GradCAM (no CAA, no CGL) ─────────────────
python stage2_cam/generate_pseudomasks.py \
  --config configs/stage2_cam.yaml \
  --use_damp_features true \
  --use_softmax_gradcam true \
  --use_caa false \               # ← no refinement
  --output_dir pseudo_masks/b3_damp_softmax/

python stage3_seg/train_seg.py \
  --pseudo_mask_dir pseudo_masks/b3_damp_softmax/ \
  --use_cgl false \               # ← no CGL
  --output_dir checkpoints/b3_damp_softmax/

python tools/eval_crossdomain.py --ckpt checkpoints/b3_damp_softmax/best.pth \
  --output_csv results/b3_damp_softmax.csv


# ── Ours: DAMP + CLIP-ES (full) ──────────────────────────────────────────
python stage2_cam/generate_pseudomasks.py \
  --config configs/stage2_cam.yaml \
  --use_damp_features true \
  --use_softmax_gradcam true \
  --use_caa true \
  --output_dir pseudo_masks/ours/

python stage3_seg/train_seg.py \
  --pseudo_mask_dir pseudo_masks/ours/ \
  --use_cgl true \
  --output_dir checkpoints/ours/

python tools/eval_crossdomain.py --ckpt checkpoints/ours/best.pth \
  --output_csv results/ours.csv
```

---

## 8. Ablation Experiments

| Ablation ID | Thay đổi | Flag |
|-------------|----------|------|
| A1 | No DAMP (CLIP-ES raw) | `--use_damp_features false` |
| A2 | DAMP + vanilla GradCAM | `--use_softmax_gradcam false` |
| A3 | No background suppression | `--use_bg_suppression false` |
| A4 | No synonym fusion | `--use_synonym_fusion false` |
| A5 | No CAA | `--use_caa false` |
| A6 | No CGL | `--use_cgl false` |
| A7 | No CRF | `--use_crf false` |

Mỗi ablation chỉ thay đổi **một flag**, giữ nguyên tất cả còn lại.

```bash
for ablation in A1 A2 A3 A4 A5 A6 A7; do
  python run_ablation.py --id $ablation --output_csv results/ablation_${ablation}.csv
done
```

---

## 9. Expected Results Table

Sau khi chạy xong, bảng kết quả nên trông như thế này.

### Pseudo-Mask Quality (Hist → BCSS, train split)

| Method | CAM mIoU | CAM+CAA mIoU | +CRF mIoU |
|--------|----------|--------------|-----------|
| CLIP-ES (raw) | ~55–60 | ~63–68 | ~68–72 |
| DAMP + vanilla GradCAM | TBD | TBD | TBD |
| DAMP + CLIP-ES (ours) | TBD | TBD | TBD |

### Final Segmentation (test split)

| Method | mIoU | FwIoU | ACC |
|--------|------|-------|-----|
| D-CAM (prior art) | ~47 | ~47 | ~64 |
| CLIP-ES baseline | TBD | TBD | TBD |
| DAMP + vanilla GradCAM | TBD | TBD | TBD |
| **DAMP + CLIP-ES (ours)** | **TBD** | **TBD** | **TBD** |

> Giá trị D-CAM tham chiếu từ Table 1 trong paper (BCSS → Hist: mIoU 47.09).

---

## 10. Logging & Visualization

```bash
# WandB logging (recommended)
wandb init --project damp_es

# Visualize CAM, pseudo mask, final prediction
python tools/visualize.py \
  --image_path data/CrossDomainSeg/BCSS/images/sample_001.png \
  --cam_dir pseudo_masks/ours/cams/ \
  --pseudo_dir pseudo_masks/ours/ \
  --pred_dir checkpoints/ours/preds/ \
  --output_dir visualizations/sample_001/

# Output: image | CAM heatmap | pseudo mask | final prediction | GT
```

---

## 11. Quick Run Checklist
[] Clone DAMP, CLIP-ES repos
[] Cài đặt env và verify import clip OK
[] Download VOC2012 + SBD → chạy CLIP-ES nguyên bản lấy mốc
[] Download Hist/BCSS/WSSS datasets
[] Chạy Stage 1 DAMP trên Office-Home để verify backbone
[] Hook DAMP features vào Stage 2 pipeline
[] Chạy Stage 2: generate pseudo masks trên BCSS (target)
[] Kiểm tra pseudo-mask quality bằng tools/eval_miou.py
[] Chạy Stage 3: train DeepLab trên pseudo masks
[] Evaluate trên BCSS test split → lấy mIoU/FwIoU/ACC
[] Chạy 3 baseline + ablation A1–A7
[] Export bảng kết quả CSV + visualization

text

---

## 12. Troubleshooting

| Vấn đề | Nguyên nhân thường gặp | Fix |
|--------|------------------------|-----|
| CAM tất cả là 0 | Class token không bị thay bằng avg patches | Bật `replace_cls_with_avg: true` |
| CAM chỉ focus vào 1 vùng nhỏ | Không dùng softmax-GradCAM | Bật `use_softmax_gradcam: true` |
| Pseudo mask nhiều vùng ignore | Confidence threshold quá cao | Giảm `confidence_threshold` từ 0.25 xuống 0.15 |
| DAMP training loss không xuống | LR quá cao hoặc batch quá nhỏ | Thử `lr: 5e-5`, `batch_size: 64` |
| CRF chạy chậm | pydensecrf không có CUDA | Dùng multiprocessing: `--n_workers 8` |
| OOM ở Stage 3 | Batch quá lớn với ResNet101 | Giảm `batch_size: 4`, bật `gradient_accumulation: 2` |