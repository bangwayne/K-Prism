import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random

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
            img = np.clip(img, -991, 500)
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

dataset_info = ('amos_ct', 'ct')
source_path = '/research/cbim/medical/bg654/verse_dataset/AMOS_CT/rescale_data'
target_path = "/research/cbim/medical/bg654/verse_dataset/AMOS_CT/Data/"
process_3d_data(dataset_info, source_path, target_path, file="test")
# train_source_path = source_path + '/train'
# test_source_path = source_path + '/test'