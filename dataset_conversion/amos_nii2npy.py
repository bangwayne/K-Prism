import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
import random

# def pad(img, lab):
#     z, y, x = img.shape
#     # pad if the image size is smaller than trainig size
#     if z < 128:
#         diff = int(math.ceil((128. - z) / 2))
#         img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
#         lab = np.pad(lab, ((diff, diff), (0, 0), (0, 0)))
#     if y < 128:
#         diff = int(math.ceil((128. - y) / 2))
#         img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
#         lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0)))
#     if x < 128:
#         diff = int(math.ceil((128. - x) / 2))
#         img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
#         lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff)))
#
#     return img, lab


dataset_list = [
    ('amos_ct', 'ct'),
    # ('amos_mr', 'mr'),
    # ('BTCV', 'ct'),
    # ('structseg_oar', 'ct'),
    # ('lits', 'ct'),
    # ('kits', 'ct'),
    # ('chaos', 'mr'),
    # ('structseg_head_oar', 'ct'),
    # ('mnm', 'mr'),
    # ('brain_structure', 'mr'),
    # ('autopet', 'pet'),
]

# source_path = '/research/cbim/vast/bg654/Desktop/jupyproject/imask2former/datasets/BTCV/rescale_data'
# target_path = 'npy'

# for dataset, modality in dataset_list:
#
#     shutil.copytree(os.path.join(source_path, dataset, 'list'), os.path.join(target_path, dataset, 'list'))
#     for name in os.listdir(os.path.join(source_path, dataset)):
#         if 'gt' in name:
#             idx = name.split('_gt')[0]
#
#             img = sitk.ReadImage(os.path.join(source_path, dataset, f"{idx}.nii.gz"))
#             img = sitk.GetArrayFromImage(img).astype(np.float32)
#             lab = sitk.ReadImage(os.path.join(source_path, dataset, f"{idx}_gt.nii.gz"))
#             lab = sitk.GetArrayFromImage(lab).astype(np.int8)
#
#             if modality == 'ct':
#                 img = np.clip(img, -991, 500)
#             else:
#                 percentile_2 = np.percentile(img, 2, axis=None)
#                 percentile_98 = np.percentile(img, 98, axis=None)
#                 img = np.clip(img, percentile_2, percentile_98)
#
#             mean = np.mean(img)
#             std = np.std(img)
#
#             img -= mean
#             img /= std
#
#             # img, lab = pad(img, lab)
#
#             if dataset == 'amos_mr':
#                 lab[lab == 14] = 0
#                 lab[lab == 15] = 0
#
#             img, lab = img.astype(np.float32), lab.astype(np.int8)
#
#             # np.save(os.path.join(target_path, dataset, f"{idx}.npy", img)
#             # np.save(os.path.join(target_path, dataset, f"{idx}_gt.npy", lab)
#
#             print(name)
import yaml
# Load YAML file
with open("/research/cbim/medical/bg654/verse_dataset/AMOS/amos_ct_3d/list/dataset.yaml", "r") as file:
    data = yaml.safe_load(file)  # Use safe_load to avoid security issues

# Print the data
print(data)
source_path = '/research/cbim/medical/bg654/verse_dataset/AMOS/amos_ct_3d'
# output_image_path = '/research/cbim/medical/bg654/verse_dataset/AMOS/rescale_data/train/image'
# output_gt_path = '/research/cbim/medical/bg654/verse_dataset/AMOS/rescale_data/train/Annotations'
# os.makedirs(output_image_path, exist_ok=True)
# os.makedirs(output_gt_path, exist_ok=True)
# for i in data:
#     print(i)
#     train_image_path = source_path + "/" + str(i) + ".nii.gz"
#     train_gt_path = source_path + "/" + str(i) + "_gt.nii.gz"
#     shutil.move(train_image_path, output_image_path)
#     shutil.move(train_gt_path, output_gt_path)


# output_image_folder = '/research/cbim/medical/bg654/verse_dataset/AMOS/rescale_data/test/image'  # Change this to your image destination folder
# output_gt_folder = '/research/cbim/medical/bg654/verse_dataset/AMOS/rescale_data/test/Annotations'  # Change this to your ground truth destination folder
#
# # Create destination folders if they don't exist
# os.makedirs(output_image_folder, exist_ok=True)
# os.makedirs(output_gt_folder, exist_ok=True)
#
# # Iterate over all files in the source folder
# for filename in os.listdir(source_path):
#     file_path = os.path.join(source_path, filename)
#
#     # Check if it's a file
#     if os.path.isfile(file_path):
#         if filename.endswith("_gt.nii.gz"):
#             shutil.move(file_path, os.path.join(output_gt_folder, filename))  # Move to gt folder
#         elif filename.endswith(".nii.gz"):
#             shutil.move(file_path, os.path.join(output_image_folder, filename))  # Move to image folder
#
# print("Files have been successfully moved!")

source_path = '/research/cbim/medical/bg654/verse_dataset/AMOS/rescale_data'
train_source_path = source_path + '/train'
test_source_path = source_path + '/test'

#

for dataset, modality in dataset_list:
    for name in os.listdir(os.path.join(test_source_path, "image")):
        # if 'gt' in name:
        idx = name.split('.nii.gz')[0]

        img = sitk.ReadImage(os.path.join(test_source_path, "image", f"{idx}.nii.gz"))
        img = sitk.GetArrayFromImage(img).astype(np.float32)
        lab = sitk.ReadImage(os.path.join(test_source_path, "Annotations", f"{idx}_gt.nii.gz"))
        lab = sitk.GetArrayFromImage(lab).astype(np.int8)

        if modality == 'ct':
            img = np.clip(img, -991, 500)
        else:
            percentile_2 = np.percentile(img, 2, axis=None)
            percentile_98 = np.percentile(img, 98, axis=None)
            img = np.clip(img, percentile_2, percentile_98)

        mean = np.mean(img)
        std = np.std(img)
        print(mean, std)
        # img -= mean
        # img /= std

        nonzero_slices = np.where(np.any(lab > 0, axis=(1, 2)))[0]  # 找到有标签的层索引

        # if len(nonzero_slices) == 0:
        #     print(f"Skipping {name}: No labeled slices found.")
        #     continue  # 如果没有标签，跳过该样本
        #
        # # **Step 4: 随机选取最多 30 层**
        selected_slices = random.sample(list(nonzero_slices), min(30, len(nonzero_slices)))

        # **Step 5: 只保留这些层**
        img = img[selected_slices, :, :]
        lab = lab[selected_slices, :, :]

        # img, lab = pad(img, lab)

        # if dataset == 'amos_mr':
        #     lab[lab == 14] = 0
        #     lab[lab == 15] = 0

        img, lab = img.astype(np.float32), lab.astype(np.int8)
        print(img.shape)

        img_sitk = sitk.GetImageFromArray(img)
        lab_sitk = sitk.GetImageFromArray(lab.astype(np.uint8))  # 确保 label 以 uint8 保存

        target_dir = "/research/cbim/medical/bg654/verse_dataset/AMOS/AMOS_CT/Data/train"
        img_save_path = os.path.join(target_dir, "image", f"{idx}.nii.gz")
        lab_save_path = os.path.join(target_dir, "Annotations", f"{idx}_gt.nii.gz")

        sitk.WriteImage(img_sitk, img_save_path)
        sitk.WriteImage(lab_sitk, lab_save_path)
        #
        # # np.save(os.path.join(target_path, dataset, f"{idx}.npy", img)
        # # np.save(os.path.join(target_path, dataset, f"{idx}_gt.npy", lab)
        #
        # print(name)