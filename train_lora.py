
import torch
import os
import csv
import copy
from peft import get_peft_model
from monai.utils import set_determinism
from config import parse_args

from saving_utils import get_model_directory_and_name, save_and_print_selected_hyperparameters
from train_val import train_one_epoch, validate, EarlyStopping
from data.dataset import load_train_val_data
from models.loss import PartialLogLikelihood
from models.model_utils import build_model, load_pretrained_backbone, get_lora_config, print_trainable_layers



def create_model_lora(config, device):
    # Initialize model and load pretrained CT-CLIP weights
    model, tokenizer = build_model(config.modality, config.deepsurv_hidden_layers, config.deepsurv_dropout_prob, device)
    load_pretrained_backbone(model, config.modality, config.pretrained_weights_dir)

    lora_config = get_lora_config(config.freeze_option, config.modality)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # print_trainable_layers(model)
            
    model.to(device) 

    return model, tokenizer



def train_model_lora(model, tokenizer, model_dir, model_name, training_dataloader, validation_dataloader, device, config):

    # Define loss and optimizer
    loss_fn = PartialLogLikelihood()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=config.weight_decay)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, delta=0.)

    best_c_index = 0
    model_filepath = os.path.join(model_dir, model_name) 
    print(f"\nStart training...")

    # Write progress to csv file
    csv_fpath = model_filepath+'.csv' 
    print(f"Save progress in csv file: {csv_fpath}")
    with open(csv_fpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "TrainLoss", "TrainCindex", "ValLoss", "ValCindex"])


    for epoch in range(1, config.epochs+1):

        train_loss, train_c_ind = train_one_epoch(training_dataloader, model, tokenizer, loss_fn, optimizer, device, config.modality)
        val_loss, val_c_ind = validate(validation_dataloader, model, tokenizer, loss_fn, device, config.modality)
        
        print(f"Epoch: {epoch}\t Train loss: {train_loss:.4f}\t Train C-index: {train_c_ind:.4f}\t Val loss: {val_loss:.4f}\t Val C-index: {val_c_ind:.4f}")

        if val_c_ind > best_c_index:
            best_c_index = val_c_ind

            print(f"Save model for epoch {epoch}")
            model_copy = copy.deepcopy(model)
            merged_model = model_copy.merge_and_unload()
            torch.save(merged_model.state_dict(), model_filepath+'_best.pth')

        with open(csv_fpath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_c_ind, val_loss, val_c_ind])

        if early_stopping.early_stop(val_loss):
            print("Early stopping")
            break

    print("Finished training")




def main():
    config = parse_args()

    set_determinism(seed=config.seed)

    model_dir, model_name = get_model_directory_and_name(config, lora=True)
    save_and_print_selected_hyperparameters(config, model_name, model_dir)

    training_dataloader, validation_dataloader = load_train_val_data(config)

    device = (config.cuda_choice if torch.cuda.is_available() else "cpu")
    print(f"\nUsing {device} device")

    model, tokenizer = create_model_lora(config, device)
    train_model_lora(model, tokenizer, model_dir, model_name, training_dataloader, validation_dataloader, device, config)




if __name__ == "__main__":
    main()

