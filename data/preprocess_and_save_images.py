# This script can be used to preprocess and save dicom images

import os
import torch
from utils_preprocessing import ct_clip_image_preprocessing_pipeline


dicom_dir = "/path/to/folder/containing/dicom/files"
preprocessed_images_save_dir = "/path/to/saving/folder/for/prepreprocessed/images"


# Loop though all dicom volumes and save a preprocessed version 
list_saved_dicoms = os.listdir(dicom_dir)
list_saved_dicoms.sort()

for img_name in list_saved_dicoms:
    print(img_name)
    img_path = os.path.join(dicom_dir, img_name)

    preprocessed_image = ct_clip_image_preprocessing_pipeline(img_path)

    img_save_dir = os.path.join(preprocessed_images_save_dir, img_name+".pt")
    print(f"Save path: {img_save_dir}\n")
    torch.save(preprocessed_image, img_save_dir) 

