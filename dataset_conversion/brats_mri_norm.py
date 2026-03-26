import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random
import yaml
import os
import shutil
from collections import defaultdict

def process_3d_data(dataset_info, source_path, target_path, file="train"):
    dataset, modality = dataset_info
    if file == "train":
        file_path = os.path.join(source_path, 'train')
    else:
        file_path = os.path.join(source_path, 'test')

    image_dir = os.path.join(file_path, "image")
    annotation_dir = os.path.join(file_path, "annotations")

    patient_dict = defaultdict(dict)

    # Group files by patient ID
    for fname in os.listdir(image_dir):
        if not fname.endswith(".nii.gz"):
            continue
        full_path = os.path.join(image_dir, fname)
        name = fname.replace(".nii.gz", "")
        if name.endswith("_seg"):
            continue  # skip labels

        parts = name.split("_")
        patient_id = "_".join(parts[:-1])
        modality_name = parts[-1]
        patient_dict[patient_id][modality_name] = full_path

    for patient_id, modalities in patient_dict.items():
        required_modalities = ["t1", "t2", "flair", "t1ce"]
        if not all(mod in modalities for mod in required_modalities):
            print(f"Skipping {patient_id}: missing one or more modalities.")
            continue

        label_path = os.path.join(annotation_dir, f"{patient_id}_gt.nii.gz")
        if not os.path.exists(label_path):
            print(f"Skipping {patient_id}: label file not found.")
            continue

        for mod in required_modalities:
            img = sitk.ReadImage(modalities[mod])
            # print(modalities[mod])
            img_np = sitk.GetArrayFromImage(img).astype(np.float32)
        # Load one modality for processing (e.g., t2 here)
        # img = sitk.ReadImage(modalities["t2"])
        # img_np = sitk.GetArrayFromImage(img).astype(np.float32)
            lab = sitk.ReadImage(label_path)
            lab_np = sitk.GetArrayFromImage(lab).astype(np.int8)

            img_min = img_np.min()
            img_max = img_np.max()
            if img_max > img_min:
                img_np = (img_np - img_min) / (img_max - img_min) * 255.0
            else:
                img_np = np.zeros_like(img_np)

            nonzero_slices = np.where(np.any(lab_np > 0, axis=(1, 2)))[0]
            if len(nonzero_slices) == 0:
                print(f"Skipping {patient_id}: No labeled slices.")
                continue
            save_name = modalities[mod].split("/")[-1]
            img_np = img_np[nonzero_slices, :, :]
            lab_np = lab_np[nonzero_slices, :, :]

            img_sitk = sitk.GetImageFromArray(img_np.astype(np.float32))
            lab_sitk = sitk.GetImageFromArray(lab_np.astype(np.uint8))

            save_dir = os.path.join(target_path, file)
            os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)

            sitk.WriteImage(img_sitk, os.path.join(save_dir, "image", save_name))
            sitk.WriteImage(lab_sitk, os.path.join(save_dir, "annotations", f"{patient_id}_gt.nii.gz"))

        print(f"✅ Processed {patient_id}, shape: {img_np.shape}")


# 路径设置
def sort_and_rename_files(source_dir, image_dir, annotation_dir):
    """
    - Rename files containing 'seg' to use 'gt' instead.
    - Move 'gt' files to annotation_dir.
    - Move all others to image_dir.
    """
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    for file in os.listdir(source_dir):
        if not file.endswith(".nii.gz"):
            continue

        src_path = os.path.join(source_dir, file)

        if "seg" in file.lower():
            new_name = file.replace("seg", "gt")
            dst_path = os.path.join(annotation_dir, new_name)
            print(f"Renaming & moving: {file} → {new_name} → annotation/")
        else:
            dst_path = os.path.join(image_dir, file)
            print(f"Moving image: {file} → image/")

        shutil.copy(src_path, dst_path)

def copy_yaml_file(yaml_path, source_dir, target_path):
    # yaml_path = "config.yaml"
    # source_dir = "/path/to/source"  # 原始数据存放路径
    os.makedirs(target_path, exist_ok=True)
    target_image_dir = os.path.join(target_path, "image")
    target_label_dir = os.path.join(target_path, "annotations")

    # 确保输出文件夹存在
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    # 读取 YAML 文件
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # print(cfg)
    patient_list = cfg["patients"]

    for name in patient_list:
        file_path = os.path.join(source_dir, name)
        sort_and_rename_files(file_path, target_image_dir, target_label_dir)


def create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="."):
    """
    从 source_path 下获取所有子文件夹名，划分为 train/test，并保存为 YAML 文件。

    Args:
        source_path (str): 数据的根目录，每个子文件夹是一个样本。
        train_ratio (float): 训练集占比（默认 0.8）。
        seed (int): 随机种子（保证可复现）。
        save_dir (str): YAML 文件的保存目录。
    """
    all_folders = [d for d in os.listdir(source_path)
                   if os.path.isdir(os.path.join(source_path, d))]

    all_folders.sort()  # 可选：保持固定顺序
    random.seed(seed)
    random.shuffle(all_folders)

    split_idx = int(len(all_folders) * train_ratio)
    train_list = all_folders[:split_idx]
    test_list = all_folders[split_idx:]

    train_yaml_path = os.path.join(save_dir, "train.yaml")
    test_yaml_path = os.path.join(save_dir, "test.yaml")

    with open(train_yaml_path, 'w') as f:
        yaml.dump({'patients': train_list}, f)
    with open(test_yaml_path, 'w') as f:
        yaml.dump({'patients': test_list}, f)

    print(f"Saved train list ({len(train_list)} samples) → {train_yaml_path}")
    print(f"Saved test list ({len(test_list)} samples) → {test_yaml_path}")


# folder_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/MICCAI_BraTS2020_TrainingData"
# list_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/list"
# os.makedirs(list_path, exist_ok=True)
# create_yaml(folder_path, save_dir=list_path)
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/list/test.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/MICCAI_BraTS2020_TrainingData"
# target_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/Data/test"
# copy_yaml_file(yaml_path, source_path, target_path)
#
dataset_info = ('brats_mri', 'mri')
source_path = '/research/cbim/medical/bg654/verse_dataset/BraTS2020/rescale_data'
target_path = "/research/cbim/medical/bg654/verse_dataset/BraTS2020/Data/"
process_3d_data(dataset_info, source_path, target_path, file="train")
process_3d_data(dataset_info, source_path, target_path, file="test")
