# K-Prism

**K-Prism** is a medical image segmentation framework that combines point-based interactive segmentation with reference-guided (one-shot) learning. It supports 2D and 3D medical images across multiple modalities and anatomical structures.

## Architecture Overview

```
Input Image ──► Backbone (UTNet) ──► Multi-scale Features (res3/res4/res5)
                                            │
                          ┌─────────────────┴──────────────────┐
                          │                                     │
               Reference Images (optional)              Point Prompts
               MaskEncoder + ObjectSummarizer         PromptEncoder
                          │                                     │
                          └──────────────┬──────────────────────┘
                                         │
                          MultiScaleMaskedTransformerDecoder
                               (with Mixture-of-Experts)
                                         │
                                   Mask Predictions
```

**Key components:**

| Module | File | Description |
|--------|------|-------------|
| `KPrism` | `kprism/KPrism_model.py` | Base model class |
| `KPrismTrainWrapper` | `kprism/Trainwrapper.py` | Training wrapper with 3 modes |
| `InferenceCore` | `kprism/inference/inference.py` | Iterative inference |
| `UTNet` | `kprism/modeling/backbone/utnet.py` | U-Net backbone with transformer blocks |
| `SegHead` | `kprism/modeling/meta_arch/seghead.py` | Segmentation head (PixelFuser + decoder) |
| `MaskEncoder` | `kprism/modeling/task_encoder/mask_encoder.py` | Encodes reference image + mask pairs |
| `ObjectSummarizer` | `kprism/modeling/task_encoder/object_summarier.py` | Aggregates object-level support features |
| `PointSampler` | `kprism/utils/point_sampler.py` | Generates and refines point prompts |

## Data Structures

### Input batch (`batched_inputs`)
A list of dicts, one per image:

```python
{
    "image":        Tensor (C, H, W),          # query image
    "target": {
        "labels":   Tensor (N,),               # class indices
        "masks":    Tensor (N, H, W),          # binary masks
    },
    "q_index":      int,                       # query class index
    "ref_img":      Tensor (num_ref, 3, H, W), # reference images
    "ref_mask":     Tensor (num_ref, 1, H, W), # reference masks
    "epoch":        int,                       # current epoch
    "size_info":    (H, W),                    # original size before padding
    "pad_info":     tuple,                     # padding applied
    "scale_factor": float,                     # resize scale factor
}
```

### Point representation
```python
points_list = [
    (coords, labels)   # per mask
]
# coords: Tensor (N, 2)  — [y, x] format
# labels: Tensor (N,)    — 1 = positive, 0 = negative
```

### Feature maps
Multi-scale feature dict returned by the backbone:
```python
{
    "res3": Tensor (B, 96,  H/8,  W/8),
    "res4": Tensor (B, 192, H/16, W/16),
    "res5": Tensor (B, 384, H/32, W/32),
}
```

### Inference output
```python
result_dict = {
    0: [Tensor (1, H, W), ...],  # masks after iteration 0
    1: [Tensor (1, H, W), ...],  # masks after iteration 1
    ...
}
point_dict = {
    0: [(coords, labels), ...],  # points used at iteration 0
    ...
}
```

## Training

### Configuration
Edit `kprism/config/train_config.yaml` to set your paths and hyperparameters, and `kprism/config/data/datasets.yaml` to configure datasets.

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.iter_num` | 3 | Click refinement iterations per sample |
| `training.training_click_mode` | `['1','2','3']` | Active training modes |
| `training.sampling_probs` | `[0.3, 0.4, 0.3]` | Probability per mode |
| `training.click_loss_weight` | 0.8 | Loss weight for refinement iterations |
| `training.epochs` | 75 | Total training epochs |
| `training.batch_size` | 4 | Batch size per GPU |
| `solver.base_lr` | 1e-4 | Base learning rate |

**Training modes:**
- **Mode 1** (prob 0.3): Query-based segmentation with iterative point refinement
- **Mode 2** (prob 0.4): Reference-guided segmentation with iterative point refinement
- **Mode 3** (prob 0.3): Point-only segmentation (no initial query)

### Launch training
```bash
bash run_train_moe.sh
```

Or directly:
```bash
python muti_gpu_train_Verse_Cardiac_final.py --config_file kprism/config/train_config.yaml
```

## Inference

### Configuration
Edit `kprism/config/eval_config.yaml`:
- Set `testing.resume` to your checkpoint path
- Set `dataset.dataset_path` to your data directory
- Set `dataset.dataset_name` to the target dataset (e.g. `"ACDC"`)

Key inference parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `testing.iter_num` | 5 | Click refinement iterations |
| `testing.testing_click_mode` | `['3']` | Inference mode |
| `testing.num_ref` | 2 | Number of reference images |

### Run evaluation
```bash
bash run_eval_moe.sh
```

Or in Python:
```python
from hydra import compose, initialize
from kprism.inference.inference import InferenceCore

with initialize(config_path="kprism/config"):
    cfg = compose(config_name="eval_config")

model = InferenceCore(cfg)
checkpoint = torch.load(cfg.testing.resume, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval().cuda()

# batched_inputs: list of dicts (see Data Structures above)
result_dict, point_dict = model(batched_inputs)
# result_dict[iter] -> list of predicted masks (sigmoid applied)
```

## Supported Datasets

The framework ships with dataset configs for:

| Dataset | Modality | Classes |
|---------|----------|---------|
| AMOS_MRI / AMOS_CT | 3D CT/MRI | 15 abdominal organs |
| MM / ACDC | 3D cardiac MRI | RV, Myocardium, LV |
| LITS | 3D CT | Liver tumor |
| KITS | 3D CT | Kidney tumor |
| ISIC | 2D dermoscopy | Skin lesion |
| BKAI_POLY | 2D endoscopy | Polyps |
| BraTS | 2D MRI | Brain tumor |
| PAPILA | 2D fundus | Optic disc |
| KPI | 2D pathology | Glomeruli |

To add a new dataset, add an entry to `kprism/config/data/datasets.yaml` following the existing format.

## Dependencies

- PyTorch ≥ 1.12
- detectron2
- hydra-core
- omegaconf
- SimpleITK
- fvcore
- tensorboardX
- tqdm
- opencv-python
