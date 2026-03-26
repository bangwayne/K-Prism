import os
import random
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
import json  # or use yaml
import os
import random
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
import json


import os
import random
import numpy as np
import SimpleITK as sitk
import json


def build_reference_slice_dict(
    mask_dir,
    mode="one-shot",  # only metadata use, does not affect behavior
    num_patients=10,
    save_path="ref_slice_dict.json",
    seed=42
):
    """
    For each randomly selected patient, find one slice that contains all labels.
    If no such slice exists, fallback to the one with most labels.

    Args:
        mask_dir (str): Path to mask files (e.g., *_gt.nii.gz).
        mode (str): Metadata indicator ("one-shot" or "five-shot").
        num_patients (int): Number of patients to include.
        save_path (str): Output JSON path.
        seed (int): Random seed.
    """
    random.seed(seed)
    all_files = sorted([f for f in os.listdir(mask_dir) if f.endswith("_gt.nii.gz")])
    all_patients = [f.split("_gt")[0] for f in all_files]

    sampled_patients = random.sample(all_patients, k=min(num_patients, len(all_patients)))
    ref_dict = {}

    for patient_id in sampled_patients:
        mask_path = os.path.join(mask_dir, f"{patient_id}_gt.nii.gz")
        if not os.path.isfile(mask_path):
            continue

        itk_mask = sitk.ReadImage(mask_path)
        np_mask = sitk.GetArrayFromImage(itk_mask)  # [D, H, W]

        label_ids = np.unique(np_mask)
        all_labels = set(label_ids[label_ids != 0])
        if not all_labels:
            print(f"[Skip] Patient {patient_id} has only background.")
            continue

        # Step 1: for each slice, store the label set
        slice_label_map = []
        for z in range(np_mask.shape[0]):
            unique_labels = np.unique(np_mask[z])
            label_set = set(unique_labels[unique_labels != 0])
            slice_label_map.append(label_set)

        # Step 2: find slice(s) that contain all labels
        candidate_slices = [i for i, s in enumerate(slice_label_map) if all_labels.issubset(s)]

        if candidate_slices:
            selected_slice = random.choice(candidate_slices)
        else:
            slice_label_counts = [len(s & all_labels) for s in slice_label_map]
            selected_slice = int(np.argmax(slice_label_counts))
            print(f"[Info] Patient {patient_id}: no full-label slice, fallback to most-covering slice.")

        ref_dict[patient_id] = {
            "mode": mode,
            "slice_index": int(selected_slice),
            "labels": [f"label_{int(l)}" for l in sorted(all_labels)]
        }

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(ref_dict, f, indent=2)

    print(f"[✓] Saved reference dict ({mode}) with {len(ref_dict)} patients to {save_path}")
    return ref_dict

build_reference_slice_dict(
    mask_dir="/research/cbim/medical/bg654/verse_dataset/ACDC/Data/test/annotations",
    mode="one-shot",
    num_patients=1,
    save_path="/research/cbim/medical/bg654/verse_dataset/ACDC/Data/test/one_shot_refs.json"
)