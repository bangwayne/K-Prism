import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random
import yaml
import imageio
from PIL import Image

import os
import shutil

import os
import numpy as np
from PIL import Image
from collections import Counter
import glob

def rename_files_and_analyze_mask(img_dir, mask_dir):
    renamed_img_dir = os.path.join(img_dir, "renamed")
    renamed_mask_dir = os.path.join(mask_dir, "renamed")
    os.makedirs(renamed_img_dir, exist_ok=True)
    os.makedirs(renamed_mask_dir, exist_ok=True)

    total_area_counter = Counter()

    for filename in os.listdir(mask_dir):
        if not filename.endswith(".png") or "mask" not in filename:
            continue

        # ========== Step 1: rename ==========
        name_split = filename.replace(".png", "").split("_")
        id_part = "_".join(name_split[:-2])  # 08_373_01_5_5120_0
        slice_id = name_split[-1]  # '3'

        # Mask rename → xxx_3_gt.png
        new_mask_name = f"{id_part}_{slice_id}_gt.png"
        src_mask_path = os.path.join(mask_dir, filename)
        dst_mask_path = os.path.join(renamed_mask_dir, new_mask_name)

        # Corresponding image: ends with _img_{slice}.png
        img_filename = filename.replace("mask", "img")
        img_filename = img_filename.replace(f"_{slice_id}_gt", f"_img_{slice_id}")
        img_path = os.path.join(img_dir, img_filename)
        new_img_name = f"{id_part}_{slice_id}.png"
        dst_img_path = os.path.join(renamed_img_dir, new_img_name)

        # Rename/copy if image exists
        if os.path.exists(img_path):
            os.rename(img_path, dst_img_path)
        else:
            print(f"❗ Image file not found for mask: {filename}")

        os.rename(src_mask_path, dst_mask_path)
        print(f"✅ Renamed {filename} → {new_mask_name}")

        # ========== Step 2: area analysis ==========
        mask = np.array(Image.open(dst_mask_path))
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            total_area_counter[u] += c

    # 打印掩膜分布
    print("\n📊 掩膜值统计（像素数）:")
    for label, area in sorted(total_area_counter.items()):
        print(f"  Label {label}: {area} pixels")

    return total_area_counter

# img_dir = "/path/to/image_folder"
# mask_dir = "/path/to/mask_folder"
# rename_files_and_analyze_mask(img_dir, mask_dir)

# import imageio
# from PIL import Image
def binarize_and_analyze_masks(mask_dir):
    mask_areas = []

    # 遍历所有 png 掩膜文件
    for mask_path in glob.glob(os.path.join(mask_dir, "*.png")):
        mask = np.array(Image.open(mask_path))
        print(mask.shape)
        # 如果是 RGB，转换成灰度
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # 二值化：<128 → 0，>=128 → 255
        bin_mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        # 统计 255 的像素个数（前景）
        foreground_pixels = np.sum(bin_mask == 255)
        mask_areas.append(foreground_pixels)

        print(f"{os.path.basename(mask_path)} → Foreground pixels: {foreground_pixels}")

    # 分位数和均值统计
    mask_areas = np.array(mask_areas)
    p20 = np.percentile(mask_areas, 20)
    p50 = np.percentile(mask_areas, 50)
    mean = np.mean(mask_areas)

    print("\n📊 Area Stats Across All Masks:")
    print(f"  20% percentile: {p20:.2f}")
    print(f"  50% percentile (median): {p50:.2f}")
    print(f"  Mean foreground area: {mean:.2f}")

    return mask_areas

def process_and_filter_masks(mask_dir, image_dir, save_mask_dir, save_img_dir, min_foreground=2000):
    os.makedirs(save_mask_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    kept = 0
    removed = 0

    for mask_path in glob.glob(os.path.join(mask_dir, "*.png")):
        base_name = os.path.basename(mask_path)
        img_name = base_name.replace("_gt.png", ".png")
        img_path = os.path.join(image_dir, img_name)

        # 读取并转为灰度
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # 二值化
        bin_mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

        # 统计 255 数量
        fg_count = np.sum(bin_mask == 255)

        if fg_count >= min_foreground:
            # 保存为单通道掩膜
            Image.fromarray(bin_mask).save(os.path.join(save_mask_dir, base_name))
            # 同时复制图像文件
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(save_img_dir, img_name))
            else:
                print(f"⚠️ 图像文件不存在：{img_name}")
            kept += 1
        else:
            removed += 1
        print(base_name)

    print(f"\n✅ Done. Kept {kept} masks. Removed {removed} masks with < {min_foreground} pixels.")

def select_random_subset(image_dir, mask_dir, save_img_dir, save_mask_dir, sample_size=2000, seed=42):
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    # 找到所有 mask 文件（假设以 "_gt.png" 结尾）
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith("_gt.png")]
    mask_files.sort()  # 可选：保持一致性
    random.seed(seed)
    random.shuffle(mask_files)

    selected = mask_files[:sample_size]
    print(f"🎯 从 {len(mask_files)} 中选取了 {len(selected)} 个样本")

    for mask_file in selected:
        # 得到图像文件名
        img_file = mask_file.replace("_gt.png", ".png")

        src_mask_path = os.path.join(mask_dir, mask_file)
        src_img_path = os.path.join(image_dir, img_file)

        dst_mask_path = os.path.join(save_mask_dir, mask_file)
        dst_img_path = os.path.join(save_img_dir, img_file)

        # 拷贝文件
        shutil.copy(src_mask_path, dst_mask_path)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"⚠️ 找不到对应图像文件: {img_file}")

    print("✅ 随机选取并保存完成！")


def process_existing_2d_data(dataset_info, source_path, target_path, file="train"):
    dataset, modality = dataset_info
    file_path = os.path.join(source_path, file)

    img_dir = os.path.join(file_path, "image")
    lab_dir = os.path.join(file_path, "annotations")

    save_img_dir = os.path.join(target_path, file, "image")
    save_lab_dir = os.path.join(target_path, file, "annotations")
    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lab_dir, exist_ok=True)

    for name in os.listdir(img_dir):
        # if not name.endswith(".jpg"):
        #     continue

        base_name = name.replace(".png", "")
        img_path = os.path.join(img_dir, name)
        lab_path = os.path.join(lab_dir, f"{base_name}_gt.png")
        if not os.path.exists(lab_path):
            print(f"Missing label for {base_name}, skipping...")
            continue

        # Load 2D image and label
        img = np.array(Image.open(img_path)).astype(np.float32)
        lab = np.array(Image.open(lab_path)).astype(np.uint8)
        print(f"unique_label: {np.unique(lab)}")
        print(lab.shape)
        # Normalize image to 0-255
        # img_min, img_max = img.min(), img.max()
        # if img_max > img_min:
        #     img = (img - img_min) / (img_max - img_min) * 255.0
        # else:
        #     img = np.zeros_like(img)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # (H, W) -> (H, W, 3)
        # Optionally skip blank labels
        if np.max(lab) == 0:
            print(f"Skipping {base_name}: empty label.")
            continue
        # # Save processed slices
        imageio.imwrite(os.path.join(save_img_dir, f"{base_name}.png"), img)
        imageio.imwrite(os.path.join(save_lab_dir, f"{base_name}_gt.png"), lab)

        print(f"✅ Processed {base_name}")


import os
from PIL import Image
import numpy as np

def crop_and_downsample_to_512(image, size=1024, down_size=512):
    """
    将图像裁剪为 4 个 1024x1024 patch，并缩放到 512x512
    """
    patches = []
    positions = [(0, 0), (0, 1024), (1024, 0), (1024, 1024)]
    for y, x in positions:
        patch = image[y:y+size, x:x+size]
        patch_resized = np.array(Image.fromarray(patch).resize((down_size, down_size), resample=Image.BILINEAR))
        patches.append(patch_resized)
    return patches

def process_folder(root_path, save_root):
    os.makedirs(os.path.join(save_root, "image"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "annotations"), exist_ok=True)

    for patient_name in os.listdir(root_path):
        patient_path = os.path.join(root_path, patient_name)
        img_dir = os.path.join(patient_path, "img")
        ann_dir = os.path.join(patient_path, "mask")

        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            continue

        for fname in os.listdir(img_dir):
            if not fname.endswith(".jpg"):
                continue

            base_name = fname.replace("_img.jpg", "")
            img_path = os.path.join(img_dir, fname)
            ann_path = os.path.join(ann_dir, base_name + "_mask.jpg")

            if not os.path.exists(ann_path):
                print(f"❌ Missing annotation for {fname}")
                continue

            # 读取图像和掩膜
            img = np.array(Image.open(img_path))
            ann = np.array(Image.open(ann_path))

            # 安全检查
            assert img.shape == (2048, 2048) or img.shape == (2048, 2048, 3), f"图像尺寸不匹配：{fname}"
            assert ann.shape == (2048, 2048), f"掩膜尺寸不匹配：{fname}"

            # 裁剪为 patch
            img_patches = crop_and_downsample_to_512(img)
            ann_patches = crop_and_downsample_to_512(ann)

            for idx, (i_patch, a_patch) in enumerate(zip(img_patches, ann_patches)):
                img_save_path = os.path.join(save_root, "image", f"{base_name}_patch{idx}.png")
                ann_save_path = os.path.join(save_root, "annotations", f"{base_name}_patch{idx}_gt.png")

                Image.fromarray(i_patch).save(img_save_path)
                Image.fromarray(a_patch).save(ann_save_path)

            print(f"✅ Processed {fname} → 4 patches")

# process_folder(
#     root_path="/research/cbim/medical/bg654/KIDNEY/original_testing",
#     save_root="/research/cbim/medical/bg654/verse_dataset/KPI/resize_data/test"
# )

# mask_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data/test/select_annotations"
# duplicates = find_duplicate_masks(mask_dir)
# img_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/rescale_data/train/image"
mask_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/rescale_data/train/annotations"
# rename_files_and_analyze_mask(img_dir, mask_dir)
# binarize_and_analyze_masks(mask_dir)
# process_and_filter_masks(
#     mask_dir="/research/cbim/medical/bg654/verse_dataset/KPI/resize_data/test/annotations",
#     image_dir="/research/cbim/medical/bg654/verse_dataset/KPI/resize_data/test/image",
#     save_mask_dir="/research/cbim/medical/bg654/verse_dataset/KPI/filter_data_2/test/annotations",
#     save_img_dir="/research/cbim/medical/bg654/verse_dataset/KPI/filter_data_2/test/image",
#     min_foreground=2000
# )
# source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir=target_path)
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/list/train.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/pool_data/benign"
# target_path = "/research/cbim/medical/bg654/verse_dataset/breast_cancer/rescale_data/train"
# copy_yaml_file(yaml_path, source_path, source_path, target_path)
# organize_images_masks_rename(source_path, target_path)
# source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# create_yaml(source_path, train_ratio=0.8, seed=42, save_dir="/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list")
# yaml_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/list/test.yaml"
# source_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/CXR_png"
# mask_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/data/Lung Segmentation/masks"
# target_path = "/research/cbim/medical/bg654/verse_dataset/Chest_Xray_Masks_and_Labels/rescale_data/test"
#
# copy_yaml_file(yaml_path, source_path, mask_path, target_path)
# dataset_info = ('chest_xray', 'mri')
# source_path = "/research/cbim/medical/bg654/verse_dataset/Uwaterloo_skin_cancer/rescale_data"
# target_path = "/research/cbim/medical/bg654/verse_dataset/Uwaterloo_skin_cancer/Data/test"
# # copy_file(source_path , target_path, surname="_nm_dermquest")
# process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# image_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data/test/image"
# mask_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data/test/annotations"
# select_image_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data/test/select_image"
# select_mask_dir = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data/test/select_annotations"
# select_random_subset(image_dir, mask_dir, select_image_dir, select_mask_dir)
#
#
#
dataset_info = ('chest_xray', 'mri')
source_path = "/research/cbim/medical/bg654/verse_dataset/KPI/filter_data_2"
target_path = "/research/cbim/medical/bg654/verse_dataset/KPI/Data/"
process_existing_2d_data(dataset_info, source_path, target_path, file="train")
process_existing_2d_data(dataset_info, source_path, target_path, file="test")
# process_3d_data(dataset_info, source_path, target_path, file="train")
# process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'
