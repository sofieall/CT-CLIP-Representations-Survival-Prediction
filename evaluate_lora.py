# Simple example evaluation of one model (trained using LoRA) using Harrell's C index

import torch
import json
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
from data.dataset import MultimodalSurvDataset
from train_val import evaluate_model
from models.loss import PartialLogLikelihood
from models.model_utils import build_model
from saving_utils import get_train_args_from_model_name, get_dirs_to_saved_model_weights_and_params_dict



models_parent_dir = "/path/to/parent/dir/where/trained/models/are/saved" # CHANGE
model_name = "name_of_model_to_load" # CHANGE

images_dir = "/path/to/preprocessed/images" # CHANGE
tab_data_dir = "/path/to/csv/with/preprocessed/tabular/data" # CHANGE

cuda_choice = "cuda:0"


# Get model weight directory and saved hyperparams directory
modality, freeze_options_folder, fold = get_train_args_from_model_name(model_name, lora=True) 
weights_directory, json_hyperparams_dir= get_dirs_to_saved_model_weights_and_params_dict(models_parent_dir, modality, freeze_options_folder, fold, model_name)

print(f"\nLOAD MODEL {model_name} FROM DIRECTORY {weights_directory}")

# Load file with saved hyperparams
with open(json_hyperparams_dir, 'r') as file:
    params_dict = json.load(file)

print(f"\nLoaded model hyperparams used during training\n---------------------------------------------")
for key, val in params_dict.items():
    print(f"{key}: {val}")

# Dataset and dataloader
test_set = MultimodalSurvDataset(
    modality=modality,  
    dataset_type="test",
    img_dir=images_dir, 
    tabular_dir=tab_data_dir
    ) 
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=10) 

device = (
    cuda_choice                     
    if torch.cuda.is_available()
    else "cpu"
)
print(f"\nEVALUATION\nUsing {device} device\n")

# Initialize model and load pretrained weights
model, tokenizer = build_model(modality, params_dict['deepsurv_hidden_layers'], params_dict['deepsurv_dropout_prob'], device)
model.load_state_dict(torch.load(weights_directory, map_location="cpu"))
model.to(device) 

loss_fn = PartialLogLikelihood()

# Evaluate using Harrell's C
avg_loss, risk_predictions, survival_times, event_indicators = evaluate_model(test_dataloader, model, tokenizer, loss_fn, device, modality)
c_ind = concordance_index(survival_times, -risk_predictions, event_indicators)
print(f"\nTest c-index for fold {fold}: {c_ind}")

