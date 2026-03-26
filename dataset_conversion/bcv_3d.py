import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, ITKReDirection
import os
import random
import yaml
import copy
import pdb


def ResampleImage(imImage, imLabel, save_path, mask_save_path, name, target_spacing=(1., 1., 1.)):
    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()

    imImage = ITKReDirection(imImage, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    imLabel = ITKReDirection(imLabel, target_direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()

    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    # if not os.path.exists('%s' % (save_path)):
    #     os.mkdir('%s' % (save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(mask_save_path,exist_ok=True)

    # re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]),
    #                             interp=sitk.sitkBSpline)
    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]),
                                interp=sitk.sitkLinear)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]),
                                 interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)

    cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 20, 20])

    sitk.WriteImage(cropped_img, '%s/%s.nii.gz' % (save_path, name))
    sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz' % (mask_save_path, name))
    print(f"process {name} done!")


if __name__ == '__main__':

    src_path = '/research/cbim/medical/bg654/verse_dataset/WORD/original_data/test/'
    tgt_path = '/research/cbim/medical/bg654/verse_dataset/WORD/rescale_data/test/image'
    tgt_mask_path = '/research/cbim/medical/bg654/verse_dataset/WORD/rescale_data/test/annotations'
    name_list = os.listdir(src_path + 'image')
    name_list = [name.split('.')[0] for name in name_list]

    # if not os.path.exists(tgt_path + 'list'):
    #     os.mkdir('%slist' % (tgt_path))
    # with open("%slist/dataset.yaml" % tgt_path, "w", encoding="utf-8") as f:
    #     yaml.dump(name_list, f)
    #
    # os.chdir(src_path)

    for name in name_list:
        img_name = name + '.nii.gz'
        lab_name = name + '.nii.gz'

        img = sitk.ReadImage(src_path + 'image/%s' % img_name)
        lab = sitk.ReadImage(src_path + 'annotations/%s' % lab_name)

        ResampleImage(img, lab, tgt_path, tgt_mask_path, name, (1.5, 1.5, 1.5))
        print(name, 'done')
