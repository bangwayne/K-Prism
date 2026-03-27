
# K-Prism: A knowledge-guided and prompt integrated universal medical image segmentation model

## Abstract
Medical image segmentation is fundamental to clinical decision-making, yet ex- isting models remain fragmented. They are usually trained on single knowledge sources and specific to individual tasks, modalities, or organs. This fragmentation contrasts sharply with clinical practice, where experts seamlessly integrate diverse knowledge: anatomical priors from training, exemplar-based reasoning from ref- erence cases, and iterative refinement through real-time interaction. We present K-Prism, a unified segmentation framework that mirrors this clinical flexibility by systematically integrating three knowledge paradigms: (i) semantic priors learned from annotated datasets, (ii) in-context knowledge from few-shot reference exam- ples, and (iii) interactive feedback from user inputs like clicks or scribbles. Our key insight is that these heterogeneous knowledge sources can be encoded into a dual-prompt representation: 1-D sparse prompts defining what to segment and 2-D dense prompts indicating where to attend, which are then dynamically routed through a Mixture-of-Experts (MoE) decoder. This design enables flexible switch- ing between paradigms and joint training across diverse tasks without architectural modifications. Comprehensive experiments on 18 public datasets spanning di- verse modalities (CT, MRI, X-ray, pathology, ultrasound, etc.) demonstrate that K-Prism achieves state-of-the-art performance across semantic, in-context, and interactive segmentation settings.

## Features
* Unified Segmentation Framework: A single model that supports semantic, in-context (few-shot), and interactive medical image segmentation without task-specific architectures.

* Dual-Prompt Representation: Encodes heterogeneous knowledge into a unified prompt space: 1D sparse prompts to specify what to segment and 2D dense prompts to indicate where to attend.

* Mixture-of-Experts (MoE) Decoder: Dynamically routes prompts through expert decoders, enabling flexible switching and composition of segmentation paradigms during inference.

## Example
Below is an example visualization of our K-Prism framework:

![K-Prism Framework](figure2.jpg)

## Updates
*  [2026-03-01] Code will be publicly released by the end of March 2026.
*  [2026-01-26] Our paper is accepted by ICLR2026!
*  Code will be released upon publication

## Quick Start

### Prepare the environment.

Install required python packages: `pip install -r requirements.txt`

### Prepare datasets

Please download the original datasets from their official website.


```
└── datasets
     ├── AMOS_MRI
     │   ├── train
     │   │   ├── image
     │   │       ├── patient001.nii.gz
     │   │       └── ...
     │   │   ├── annotations
     │   │       ├── patient001_gt.nii.gz
     │   │       └── ...
     │   ├── test
     │       ├── image
     │       ├── annotations
     ├── MM
     │   ├── train
     │   └── test
```

### Training
The training configurations are under `kprism/config/train_config.yaml`. The model configurations are under `kprism/config/model`.

To train the model: `sh run_train_moe.sh`


### Evaluation
The testing configurations are under `kprism/config/eval_config.yaml`.

To evaluate the model: `sh run_eval_moe.sh`


---

## Technical Details

### Dual-Prompt Representation: 1D Sparse Tokens and 2D Dense Feature Maps

K-Prism encodes all knowledge sources into two complementary prompt types that are jointly consumed by the MoE decoder.

#### 2D Dense Feature Maps (where to attend)

Point prompts and reference-based features are first projected into 2D dense feature maps that spatially indicate regions of interest.

For click-based interaction (`kprism/modeling/point_encoder/point_feature_map_encoder.py`):
- A 2-channel map of shape `(2, H, W)` is created — channel 0 for positive clicks, channel 1 for negative clicks.
- The previous segmentation prediction is appended as a 3rd channel, giving `(3, H, W)`.
- This 3D map is then downsampled to match each feature scale (e.g. `H/8 × W/8` for `res3`).

For reference-based learning, the `MaskEncoder` encodes the concatenated reference image and mask `(4, H, W) → multi-scale features`, and `ObjectSummarizer` pools them into compact object-level descriptors per scale.

#### 1D Sparse Tokens (what to segment)

The same point coordinates are also converted to 1D sparse tokens that carry semantic identity:

1. For each point at `(y, x)`, the corresponding location in the backbone feature map is sampled at `(y / stride, x / stride)`.
2. If the sampled feature is zero (point lands on background), a 3×3 neighborhood mean is used instead.
3. The sampled feature is projected through a 2-layer MLP: `256 → 128 → 256`.
4. Label embeddings (positive / negative / padding) from `PromptEncoder` are added to encode click polarity.
5. Output: `(N, B, 256)` — N sparse tokens per batch.

For semantic and in-context modes, learnable object queries (shape `(num_queries, B, 256)`) serve as the sparse tokens instead of click-derived tokens.

The two prompt types enter the MoE decoder together: 2D dense maps supply the pixel-level cross-attention keys/values, while 1D sparse tokens act as queries.

---

### Mixture-of-Experts (MoE) Decoder

The decoder (`kprism/modeling/transformer_decoder/transformer_decoder.py`) consists of `dec_layers` stacked transformer layers, each using MoE for both cross-attention and feedforward sub-layers.

#### Architecture of one MoE layer

```
Queries (1D sparse tokens)
        │
        ▼
MoE Cross-Attention   ← pixel feature keys/values (from 2D dense maps)
        │
        ▼
Standard Self-Attention
        │
        ▼
MoE Feed-Forward Network
        │
        ▼
Pixel FFN            
```

**MoE Cross-Attention (`MoE_CrossAttentionLayer`):**
`num_experts` parallel `MultiheadAttention` modules share the same inputs. A gating network `Linear(d_model → num_experts) + softmax` computes a weight for each expert. The final output is the weighted sum:
```
output = Σ  gate_weight[i] · expert_i(query, key, value)
```

**MoE Feed-Forward (`MoE_FFNLayer`):**
`num_experts` parallel `Linear → Activation → Dropout → Linear` networks, combined identically via the softmax-weighted sum.

#### Controlling the MoE module

All MoE parameters are set in `kprism/config/model/base.yaml`:

```yaml
model:
  transformer_decoder:
    dec_layers: 6          # number of stacked transformer layers
    hidden_dim: 256        # query / key / value dimension
    dim_feedforward: 256   # inner dimension of each FFN expert
    nheads: 8              # attention heads per expert
    num_experts: 5         # number of experts in each MoE layer
    rescale: [16, 8, 4]    # pixel feature downsampling scales fed to each layer group
```

To increase model capacity, raise `num_experts` or `dec_layers`. To reduce memory, lower `num_experts` or `dim_feedforward`.

---

### Three Training Modes

K-Prism is jointly trained with three modes sampled probabilistically each iteration, controlled in `kprism/config/train_config.yaml`:

```yaml
training:
  training_click_mode: ['1', '2', '3']   # which modes to use
  sampling_probs: [0.3, 0.4, 0.3]        # probability weight per mode
  iter_num: 3                             # click refinement iterations per sample
  click_loss_weight: 0.8                  # loss weight for refinement iterations (iter > 0)
```

| Mode | Name | Prompt type | `click_mode` in decoder |
|------|------|-------------|------------------------|
| `'1'` | Semantic / Query-based | Learnable object queries + clicks | `'0'` (first), `'1'` (refinement) |
| `'2'` | In-context (few-shot) | Reference image features + clicks | `'3'` (first), `'4'` (refinement) |
| `'3'` | Click-only | Clicks only, no initial query | `'2'` throughout |

Each mode runs `iter_num` iterations. At iteration 0 the model predicts from the initial prompt; from iteration 1 onward `PointSampler.get_next_points_component()` adds a correction click at the centroid of the largest error region. The correction-iteration loss is scaled by `click_loss_weight`.

The `click_mode` string is passed directly to `MultiScaleMaskedTransformerDecoder.forward()` and selects which query construction branch executes:

```
'0' → object queries only (semantic)
'1' → object queries + click tokens (semantic + refinement)
'2' → click tokens only (interactive)
'3' → ICL (in-context) queries only
'4' → ICL queries + click tokens (in-context + refinement)
```

To train with only one mode (e.g. interactive only), set:
```yaml
training_click_mode: ['3']
sampling_probs: [1.0]
```

---

### Image Resizing and Recovery During Inference

All preprocessing and postprocessing is handled by `SegmentationPreprocessor` in `kprism/inference/resize_transform.py`.
The target size is controlled by:
```yaml
setting:
  long_side_size: 512      # resize so the longer side equals this value
  size_divisibility: 32    # pad to a multiple of this value
```

#### Point coordinate recovery

Click coordinates are also stored in padded-image space. `map_valid_points_back()` reverses them:

```python
orig_x = (x - pad_left)  / scale_factor
orig_y = (y - pad_top)   / scale_factor
```

Points with label `−1` (padding/invalid) are filtered out before mapping.

---

### Key Configuration Reference

#### `kprism/config/train_config.yaml` — training behaviour

```yaml
training:
  iter_num: 3                          # click refinement iterations per training sample
  training_click_mode: ['1','2','3']   # active training modes
  sampling_probs: [0.3, 0.4, 0.3]     # sampling probability per mode
  click_loss_weight: 0.8               # loss weight for refinement iterations
  epochs: 75
  batch_size: 4
  num_ref: 1                           # reference images per in-context sample
  work_dir: "/path/to/experiment/output"
  resume: None                         # path to checkpoint to resume from

solver:
  base_lr: 0.0001
  weight_decay: 0.0
  warmup_epochs: 10
```

#### `kprism/config/eval_config.yaml` — evaluation behaviour

```yaml
testing:
  iter_num: 5                          # click refinement iterations at test time
  testing_click_mode: ['3']            # inference mode ('1', '2', or '3')
  num_ref: 2                           # reference images for in-context mode
  resume: "/path/to/checkpoint.pth"

dataset:
  dataset_path: "/path/to/dataset"
  dataset_name: "ACDC"
  ref_mode: "one_shot"                 # reference selection strategy
```

#### `kprism/config/model/base.yaml` — architecture

```yaml
model:
  transformer_decoder:
    dec_layers: 6        # transformer depth
    num_experts: 5       # MoE experts per layer
    hidden_dim: 256      # feature dimension
    nheads: 8            # attention heads per expert

  task_encoder:
    num_queries: 2       # learnable queries per class
    dec_layers: 1        # task encoder depth

  object_summarizer:
    num_summaries: 6     # summary tokens per reference scale
    num_ref: 1           # references at train time

setting:
  long_side_size: 512           # resize target for inference preprocessor
  size_divisibility: 32         # padding granularity
  point_sample_method: "largest_component"   # 'largest_component' or 'min_dis'
```

#### `kprism/config/data/datasets.yaml` — data

```yaml
dataset_path: "/path/to/dataset"
dataset_list: ["AMOS_MRI", "AMOS_CT", "MM", ...]   # datasets included in training
dataset_weight: {"AMOS_MRI": 0.3, "MM": 0.1, ...}  # sampling weight per dataset
```

---

## Citation

```bibtex
@article{guo2025k,
  title={K-Prism: A Knowledge-Guided and Prompt Integrated Universal Medical Image Segmentation Model},
  author={Guo, Bangwei and Gao, Yunhe and Ye, Meng and Gu, Difei and Zhou, Yang and Axel, Leon and Metaxas, Dimitris},
  journal={arXiv preprint arXiv:2509.25594},
  year={2025}
}
```
