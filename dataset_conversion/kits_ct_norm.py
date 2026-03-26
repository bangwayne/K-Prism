import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random
import yaml


def create_yaml(yaml_path, train_ratio=0.8, seed=42, save_dir="."):
    """
    从 source_path 下获取所有子文件夹名，划分为 train/test，并保存为 YAML 文件。

    Args:
        source_path (str): 数据的根目录，每个子文件夹是一个样本。
        train_ratio (float): 训练集占比（默认 0.8）。
        seed (int): 随机种子（保证可复现）。
        save_dir (str): YAML 文件的保存目录。
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # print(cfg)
    patient_list = cfg
    # all_folders = [d for d in os.listdir(source_path)]
    #
    # all_folders.sort()  # 可选：保持固定顺序
    random.seed(seed)
    random.shuffle(patient_list)

    split_idx = int(len(patient_list) * train_ratio)
    train_list = patient_list[:split_idx]
    test_list = patient_list[split_idx:]

    train_yaml_path = os.path.join(save_dir, "train.yaml")
    test_yaml_path = os.path.join(save_dir, "test.yaml")

    with open(train_yaml_path, 'w') as f:
        yaml.dump({'patients': train_list}, f)
    with open(test_yaml_path, 'w') as f:
        yaml.dump({'patients': test_list}, f)

    print(f"Saved train list ({len(train_list)} samples) → {train_yaml_path}")
    print(f"Saved test list ({len(test_list)} samples) → {test_yaml_path}")


def copy_yaml_file(yaml_path, source_dir, mask_path, target_path):
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
    patient_list = cfg['patients']

    for name in patient_list:
        img_name = f"{name}.nii.gz"
        label_name = f"{name}_gt.nii.gz"

        src_img = os.path.join(source_dir, img_name)
        src_label = os.path.join(mask_path, label_name)

        tgt_img = os.path.join(target_image_dir, img_name)
        tgt_label = os.path.join(target_label_dir, f"{name}_gt.nii.gz")

        # # 拷贝图像文件
        # if os.path.exists(src_img):
        #     shutil.copy(src_img, tgt_img)
        #     print(f"Copied image: {img_name}")
        # else:
        #     print(f"Image not found: {src_img}")
        #
        # # 拷贝标注文件
        # if os.path.exists(src_label):
        #     shutil.copy(src_label, tgt_label)
        #     print(f"Copied label: {label_name}")
        # else:
        #     print(f"Label not found: {src_label}")

        if os.path.exists(src_img) and os.path.exists(src_label):
            shutil.copy(src_img, tgt_img)
            print(f"Copied image: {img_name}")
            shutil.copy(src_label, tgt_label)
            print(f"Copied label: {label_name}")


def process_3d_data(dataset_info, source_path, target_path, file="train"):
    dataset, modality = dataset_info
    if file == "train":
        file_path = source_path + '/train'
    else:
        file_path = source_path + '/test'
    # for dataset, modality in dataset_list:
    for name in os.listdir(os.path.join(file_path, "image")):
        # if 'gt' in name:
        idx = name.split('.nii.gz')[0]
        img = sitk.ReadImage(os.path.join(file_path, "image", f"{idx}.nii.gz"))
        img = sitk.GetArrayFromImage(img).astype(np.float32)
        lab = sitk.ReadImage(os.path.join(file_path, "annotations", f"{idx}_gt.nii.gz"))
        lab = sitk.GetArrayFromImage(lab).astype(np.int8)
        if modality == 'ct':
            img = np.clip(img, -200, 250)
        else:
            percentile_2 = np.percentile(img, 2, axis=None)
            percentile_98 = np.percentile(img, 98, axis=None)
            img = np.clip(img, percentile_2, percentile_98)

        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:  # 避免除以 0
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)

        # mean = np.mean(img)
        # std = np.std(img)
        #
        # img -= mean
        # img /= std

        nonzero_slices = np.where(np.any(lab > 0, axis=(1, 2)))[0]  # 找到有标签的层索引
        if len(nonzero_slices) == 0:
            print(f"Skipping {name}: No labeled slices found.")
            continue  # 如果没有标签，跳过该样本
        # **Step 4: 随机选取最多 30 层**
        # selected_slices = random.sample(list(nonzero_slices), min(30, len(nonzero_slices)))
        # **Step 5: 只保留这些层**
        img = img[nonzero_slices, :, :]
        lab = lab[nonzero_slices, :, :]
        # img, lab = pad(img, lab)

        # if dataset == 'amos_mr':
        #     lab[lab == 14] = 0
        #     lab[lab == 15] = 0
        img, lab = img.astype(np.float32), lab.astype(np.int8)
        print(img.shape)
        img_sitk = sitk.GetImageFromArray(img)
        lab_sitk = sitk.GetImageFromArray(lab.astype(np.uint8))  # 确保 label 以 uint8 保存

        save_dir = target_path + file
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)
        img_save_path = os.path.join(save_dir, "image", f"{idx}.nii.gz")
        lab_save_path = os.path.join(save_dir, "annotations", f"{idx}_gt.nii.gz")

        sitk.WriteImage(img_sitk, img_save_path)
        sitk.WriteImage(lab_sitk, lab_save_path)

    # np.save(os.path.join(target_path, dataset, f"{idx}.npy", img)
    # np.save(os.path.join(target_path, dataset, f"{idx}_gt.npy", lab)
    print(name)


dataset_info = ('kits', 'ct')
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/KITS/list/test.yaml"
# source_dir = "/research/cbim/medical/bg654/verse_dataset/KITS"
# target_path = "/research/cbim/medical/bg654/verse_dataset/KITS/rescale_data/test"
# copy_yaml_file(yaml_path, source_dir, source_dir, target_path)
# create_yaml("/research/cbim/medical/bg654/verse_dataset/KITS/list/dataset.yaml", save_dir="/research/cbim/medical/bg654/verse_dataset/KITS/list")
source_path = "/research/cbim/medical/bg654/verse_dataset/KITS/rescale_data"
target_path = "/research/cbim/medical/bg654/verse_dataset/KITS/Data/"
process_3d_data(dataset_info, source_path, target_path, file="train")
process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
