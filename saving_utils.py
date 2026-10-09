
import os
from datetime import date
import json


def get_results_dir(parent_save_dir, modality, freeze_option):

    models_folder_name = get_name_of_folder_to_save_experiments_in(freeze_option)
    results_dir = os.path.join(parent_save_dir, modality, models_folder_name)
    print(f"\nModel saved in: {results_dir}")

    if not os.path.exists(os.path.join(parent_save_dir, modality)):
        os.mkdir(os.path.join(parent_save_dir, modality))
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    return results_dir


def get_model_name(modality, freeze_option, fold_value, learning_rate, lora=False):

    params = get_str_with_fold_lr(fold_value, learning_rate)
    if lora:
        model_name = f"{str(date.today())}_{modality}_model_lora_{freeze_option}_{params}"
    else:
        model_name = f"{str(date.today())}_{modality}_model_{freeze_option}_{params}"
    print(f"Model name: {model_name}")
    return model_name


def get_model_dir(results_dir, fold_value, model_name):

    fold_dir = os.path.join(results_dir, f"fold_{fold_value}")
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)

    model_directory = os.path.join(fold_dir, model_name)
    try:
        os.mkdir(model_directory)
        print(f"Directory '{model_directory}' created successfully.\n")
    except FileExistsError:
        print(f"Directory '{model_directory}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{model_directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return model_directory


def get_model_directory_and_name(config, lora=False):

    results_dir = get_results_dir(config.save_dir, config.modality, config.freeze_option)
    model_name = get_model_name(config.modality, config.freeze_option, config.fold_value, config.learning_rate, lora)
    model_dir = get_model_dir(results_dir, config.fold_value, model_name)

    return model_dir, model_name


def save_and_print_selected_hyperparameters(config, model_name, model_dir):
    
    print("\nHYPERPARAMETER CHOICES:\n-------------------------")
    print(f"Modality: {config.modality}")
    print(f"Fold value: {config.fold_value}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Freeze options: {config.freeze_option}")
    print(f"DeepSurv dropout probability: {config.deepsurv_dropout_prob}")
    print(f"Deepsurv hidden layers: {config.deepsurv_hidden_layers}")
    print(f"Early stopping patience: {config.early_stopping_patience}")

    params_dict = {
        'modality': config.modality,
        'foldval': config.fold_value,
        'epochs': config.epochs, 
        'batch_size': config.batch_size, 
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'freeze_option': config.freeze_option,
        'deepsurv_dropout_prob': config.deepsurv_dropout_prob,
        'deepsurv_hidden_layers': config.deepsurv_hidden_layers,
        'early_stopping_patience': config.early_stopping_patience,
        'used_seed': config.seed
        } 
    with open(os.path.join(model_dir, model_name + "_params" + ".json"), "w") as outfile: 
        json.dump(params_dict, outfile)


def lr_to_str(lr):
    mantissa, exp = f"{lr:.0e}".split("e")
    return f"lr{mantissa}e{int(exp)}"


def get_str_with_fold_lr(fold_value, learning_rate):
    fold_name = f"f{fold_value}"
    lr_name = lr_to_str(learning_rate)
    return f"{fold_name}_{lr_name}"


def get_name_of_folder_to_save_experiments_in(freeze_option):

    """
    Available options multimodal model: 'trall', 'trvbds', 'trvds', 'trvvpds', 'trvvptpds', 
        'trbds', 'trbtpds', 'trbvptpds', 'trvptpds', 'trds'

    Available options image model: 'trvds', 'trvvpds', 'trvpds', 'trds'

    Available options text model: 'trbds', 'trbtpds', 'trtpds', 'trds'
    
    """

    if freeze_option == "trall":
        folder_name = "train_all"
    elif freeze_option == "trvbds":
        folder_name = "train_vit_bert_deepsurv"
    elif freeze_option == "trvds":
        folder_name = "train_vit_deepsurv"
    elif freeze_option == "trvvpds":
        folder_name = "train_vit_vp_deepsurv"
    elif freeze_option == "trvvptpds":
        folder_name = "train_vit_vp_tp_deepsurv"
    elif freeze_option == "trbds":
        folder_name = "train_bert_deepsurv"
    elif freeze_option == "trbtpds": 
        folder_name = "train_bert_tp_deepsurv"
    elif freeze_option == "trbvptpds":
        folder_name = "train_bert_vp_tp_deepsurv"
    elif freeze_option == "trvptpds":
        folder_name = "train_vp_tp_deepsurv"
    elif freeze_option == "trds": 
        folder_name = "train_deepsurv"
    elif freeze_option == "trvpds": # (used in image model)
        folder_name = "train_vp_deepsurv"
    elif freeze_option == "trtpds": # (used in text model)
        folder_name = "train_tp_deepsurv"
    else: 
        raise Exception(f"Freeze option {freeze_option} not among given options.")
    return folder_name



def get_train_args_from_model_name(model_name, lora=False):

    # models are named slightly differently depending on whether LoRA is used or not
    if lora: 
        _, modality, _, _, freeze_option, fold, _ = model_name.split('_')
    else:
        _, modality, _, freeze_option, fold, _ = model_name.split('_')

    fold = int(fold[-1])
    freeze_options_folder_name = get_name_of_folder_to_save_experiments_in(freeze_option) 

    return modality, freeze_options_folder_name, fold


def get_dirs_to_saved_model_weights_and_params_dict(models_parent_dir, modality, freeze_options_folder, fold, model_name):

    fold_directory = os.path.join(models_parent_dir, modality, freeze_options_folder, f"fold_{fold}")
    model_directory = os.path.join(fold_directory, model_name)
    model_weights_directory = os.path.join(model_directory, model_name+"_best.pth")
    json_hyperparams_dir = os.path.join(model_directory, model_name+"_params.json")

    return model_weights_directory, json_hyperparams_dir

