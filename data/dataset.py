
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from data.generate_clinical_notes import generate_notes_from_clinical_variables
import warnings


class MultimodalSurvDataset(Dataset): 
    def __init__(
        self, 
        modality, 
        dataset_type, 
        img_dir, 
        tabular_dir, 
        current_fold_val=None
    ): 

        """ 
        Args:
            modality (str): Input modality. Options are "multimodal", "image", or "text".

            dataset_type (str): Dataset split. Options are "train", "val", or "test".

            img_dir (str): Path to the directory containing preprocessed images.

            tabular_dir (str): Path to the CSV file containing patient outcome data, 
                clinical data, and fold assignments.

            current_fold_val (int, optional): Index of the fold used for validation 
                (0, 1, or 2). The remaining folds are used for training. This parameter 
                is relevant only for the training and validation splits; its value does 
                not affect the test split.
        """

        self.data = self._get_patient_data_current_fold(tabular_dir, dataset_type, current_fold_val)
        self.list_patient_indices = list(self.data.index) 
        self.modality = modality
        self.img_dir = img_dir

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return len(self.list_patient_indices) 

    def __getitem__(self, idx): 
        """ Loads and returns a sample from the dataset at the given index. """
        pat_index = self.list_patient_indices[idx]

        # Image data
        if self.modality == "multimodal" or self.modality == "image":
            filename = "CT_lung-" + f"{pat_index:03}" + ".pt"
            img_path = os.path.join(self.img_dir, filename) 
            preprocessed_image = self._load_preprocessed_image(img_path)

        # Clinical data
        if self.modality == "multimodal" or self.modality == "text":
            patient_info = self.data.loc[pat_index]
            clinical = generate_notes_from_clinical_variables(patient_info)

        # Outcome data
        event_time = self.data.at[pat_index, 'duration_days']
        event_indic = self.data.at[pat_index, 'event_indicator']

        if self.modality == "multimodal":
            return preprocessed_image, clinical, event_time, event_indic
        elif self.modality == "image":
            return preprocessed_image, event_time, event_indic
        elif self.modality == "text":
            return clinical, event_time, event_indic

    
    def _get_patient_data_current_fold(self, data_dir, data_subset, val_fold_value=None):
        
        df = pd.read_csv(data_dir)
        df.duration_days = df.duration_days.div(365)

        if data_subset == 'test': 
            return df[df.foldval == -1.0]
        
        if val_fold_value not in [0, 1, 2]:
            raise Exception("Selected val_fold_value not among options (0-2)")

        if data_subset == 'train': 
            df = df[df.foldval != val_fold_value]
            df = df[df.foldval != -1] 
            return df # return rows that are neither val nor test
        elif data_subset == 'val': 
            return df[df.foldval == val_fold_value]
        else: 
            raise Exception("Chosen dataset type not among given options ('train', 'val' or 'test')")


    def _load_preprocessed_image(self, img_path):
        """
        Load image preprocessed with preprocess_and_save_images.py.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`"
            )
            img = torch.load(img_path) 
        return img  




def load_train_val_data(config):

    train_set = MultimodalSurvDataset(
        modality=config.modality,
        dataset_type="train", 
        img_dir=config.images_dir, 
        tabular_dir=config.tabular_data_dir, 
        current_fold_val=config.fold_value
        )
    validation_set = MultimodalSurvDataset(
        modality=config.modality,  
        dataset_type="val",
        img_dir=config.images_dir, 
        tabular_dir=config.tabular_data_dir, 
        current_fold_val=config.fold_value
        ) 

    print(f"\nLength of train set: {len(train_set)}")
    print(f"Length of validation set: {len(validation_set)}")

    training_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=10)  
    validation_dataloader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=10) 

    return training_dataloader, validation_dataloader
