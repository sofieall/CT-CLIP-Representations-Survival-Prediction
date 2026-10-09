
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd, 
    ScaleIntensityRanged, 
    SpatialCropd, 
    SpatialPadd
)



def generate_load_and_preprocess_transforms_ctvit():
    """ Preprocessing transforms for CT-ViT """

    transform = Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=False,
                meta_key_postfix="meta_dict",
                reader="itkreader",
                affine_lps_to_ras=True
            ),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            Orientationd(keys=["image"], axcodes="LPS"),
            Spacingd(keys=["image"], pixdim=[0.75, 0.75, 1.5], padding_mode="border"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            )
        ]
    )
    return transform


def generate_crop_and_pad_transforms(roi_center, wanted_matrix_size, pad_value=-1):

    transform = Compose(
        [
            SpatialCropd(
                keys=["image"], 
                roi_center=roi_center,
                roi_size=wanted_matrix_size
            ),
            SpatialPadd(keys=["image"],
                spatial_size=wanted_matrix_size,
                method="symmetric",
                mode="constant",
                value=pad_value
            ), 
            EnsureTyped(keys=["image"], dtype=torch.float32)
        ]
    )
    return transform



def compute_roi_center(img_to_crop, desired_img_size):
    """
    Computes the region of interest (ROI) center based on the ratio between the 
    original image size and the desired image size.

    Note:
        The ROI center calculation may need to be adjusted for different datasets.
    """

    if img_to_crop.ndim == 4:
        img_to_crop = img_to_crop[0]
    
    dim0 = np.ceil(img_to_crop.shape[0]/2)

    if img_to_crop.shape[0] > 1.2*desired_img_size[0]:
        dim1 = np.ceil(img_to_crop.shape[1]/2)
    else:
        n_pix_to_crop = img_to_crop.shape[1] - (desired_img_size[1])
        dim1 = round(img_to_crop.shape[1] - (desired_img_size[1]/2) - n_pix_to_crop*1/4)

    if img_to_crop.shape[2] < 2*desired_img_size[2]:
        dim2 = round(img_to_crop.shape[2] - (desired_img_size[2]/2))
    else:
        n_pix_to_crop = img_to_crop.shape[2] - (desired_img_size[2])
        dim2 = round(img_to_crop.shape[2] - (desired_img_size[2]/2) - n_pix_to_crop/5)

    return (dim0, dim1, dim2)



def ct_clip_image_preprocessing_pipeline(img_path):

    goal_matrix_size = (480, 480, 240)
    load_and_preprocess = generate_load_and_preprocess_transforms_ctvit()
    preprocessed_data = load_and_preprocess({"image": img_path})
    roi_loc = compute_roi_center(preprocessed_data["image"], goal_matrix_size)
    crop_and_pad = generate_crop_and_pad_transforms(roi_loc, goal_matrix_size, pad_value=-1) # -1 is min value in image
    preprocessed_data = crop_and_pad(preprocessed_data)
    preprocessed_image = preprocessed_data["image"] 
    preprocessed_image = preprocessed_image.permute(0, 3, 1, 2)
    return preprocessed_image

