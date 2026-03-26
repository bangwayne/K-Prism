
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
*  [2026-03-26] Codebase is released.
*  [2026-01-26] Our paper is accepted by ICLR2026!
*  Code will be released upon publication

## Quick Start

### Prepare the environment.
We recommend using a clean conda environment to avoid dependency conflicts.
Please install the required Python packages first:
Install required python packages: `pip install -r requirements.txt`

### Prepare datasets

Please download the original datasets from their official websites and organize them following the structure below.

#### Important:
For dataset preprocessing, please refer to the details described in our paper. In general, the input images should be preprocessed consistently with the experimental setup in the paper, and the image intensities should be normalized to the range of 0–255 before training or evaluation. Since different datasets and modalities may have different intensity distributions and conventions, users are encouraged to carefully follow the preprocessing protocol in the paper to ensure reproducibility.


```bash
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

To train the model: `sh train.sh`


### Evaluation
The testing configurations are under `kprism/config/eval_config.yaml`.

To evaluate the model: `sh run_eval.sh`


## Citation

```bibtex
@article{guo2025k,
  title={K-Prism: A Knowledge-Guided and Prompt Integrated Universal Medical Image Segmentation Model},
  author={Guo, Bangwei and Gao, Yunhe and Ye, Meng and Gu, Difei and Zhou, Yang and Axel, Leon and Metaxas, Dimitris},
  journal={arXiv preprint arXiv:2509.25594},
  year={2025}
}
```
