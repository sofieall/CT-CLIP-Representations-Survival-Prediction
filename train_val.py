
import torch 
import numpy as np
from lifelines.utils import concordance_index



def train_one_epoch(dataloader, model, tokenizer, loss_fn, optimizer, device, modality):

    model.train()

    running_loss = 0.
    all_preds = []
    all_survival_times = []
    all_event_inds = []

    if modality == "multimodal":

        for batch, (imgs, text, survival_time, event_indicator) in enumerate(dataloader):
            
            imgs, survival_time, event_indicator= imgs.to(device), survival_time.to(device), event_indicator.to(device) 
            text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            pred = model(imgs, text_tokens)
            loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

            all_preds.append(torch.squeeze(pred)) 
            all_survival_times.append(survival_time)
            all_event_inds.append(event_indicator)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * survival_time.size(0)

    elif modality == "image":

        for batch, (imgs, survival_time, event_indicator) in enumerate(dataloader):
        
            imgs, survival_time, event_indicator= imgs.to(device), survival_time.to(device), event_indicator.to(device) 
            pred = model(imgs)  
            loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

            all_preds.append(torch.squeeze(pred)) 
            all_survival_times.append(survival_time)
            all_event_inds.append(event_indicator)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * survival_time.size(0)

    elif modality == "text":
        for batch, (text, survival_time, event_indicator) in enumerate(dataloader):
        
            survival_time, event_indicator= survival_time.to(device), event_indicator.to(device) 
            text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            pred = model(text_tokens)
            loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

            all_preds.append(torch.squeeze(pred)) 
            all_survival_times.append(survival_time)
            all_event_inds.append(event_indicator)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * survival_time.size(0) 

    else: 
        raise Exception(f"Modality {modality} not among given options")

    avg_loss = running_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds)
    survival_times = torch.cat(all_survival_times)
    event_indicators = torch.cat(all_event_inds)
    c_ind = c_index(-preds, survival_times, event_indicators)

    return avg_loss, c_ind



def validate(dataloader, model, tokenizer, loss_fn, device, modality):

    model.eval()

    running_loss = 0.
    all_preds = []
    all_survival_times = []
    all_event_inds = []

    with torch.no_grad(): 

        if modality == "multimodal":

            for batch, (imgs, text, survival_time, event_indicator) in enumerate(dataloader):
                imgs, survival_time, event_indicator = imgs.to(device), survival_time.to(device), event_indicator.to(device)
                text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                pred = model(imgs, text_tokens)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred)) 
                all_survival_times.append(survival_time)
                all_event_inds.append(event_indicator)
                running_loss += loss.item() * survival_time.size(0) 

        elif modality == "image":

            for batch, (imgs, survival_time, event_indicator) in enumerate(dataloader): 

                imgs, survival_time, event_indicator = imgs.to(device), survival_time.to(device), event_indicator.to(device)
                pred = model(imgs)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred)) 
                all_survival_times.append(survival_time)
                all_event_inds.append(event_indicator)
                running_loss += loss.item() * survival_time.size(0)

        elif modality == "text":

            for batch, (text, survival_time, event_indicator) in enumerate(dataloader): 

                survival_time, event_indicator= survival_time.to(device), event_indicator.to(device)
                text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
                pred = model(text_tokens)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred)) 
                all_survival_times.append(survival_time)
                all_event_inds.append(event_indicator)
                running_loss += loss.item() * survival_time.size(0)

        else: 
            raise Exception(f"Modality {modality} not among given options")

    avg_loss = running_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds) 
    survival_times = torch.cat(all_survival_times)
    event_indicators = torch.cat(all_event_inds)
    c_ind = c_index(-preds, survival_times, event_indicators)

    return avg_loss, c_ind


def evaluate_model(dataloader, model, tokenizer, loss_fn, device, modality):

    model.eval()

    running_loss = 0.
    all_preds = []
    all_survival_times = []
    all_event_inds = []

    with torch.no_grad():

        if modality == "multimodal":
            for batch, (imgs, text, survival_time, event_indicator) in enumerate(dataloader):

                imgs, survival_time, event_indicator = imgs.to(device), survival_time.to(device), event_indicator.to(device)
                text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                pred = model(imgs, text_tokens)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred).cpu().item()) 
                all_survival_times.append(survival_time.cpu().item())
                all_event_inds.append(event_indicator.cpu().item())
                running_loss += loss.item() * survival_time.size(0)

        elif modality == "image":

            for batch, (imgs, survival_time, event_indicator) in enumerate(dataloader): 
                imgs, survival_time, event_indicator = imgs.to(device), survival_time.to(device), event_indicator.to(device)

                pred = model(imgs)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred).cpu().item()) 
                all_survival_times.append(survival_time.cpu().item())
                all_event_inds.append(event_indicator.cpu().item())
                running_loss += loss.item() * survival_time.size(0)

        elif modality == "text":

            for batch, (text, survival_time, event_indicator) in enumerate(dataloader): 
                survival_time, event_indicator = survival_time.to(device), event_indicator.to(device)
                text_tokens = tokenizer(
                    text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                pred = model(text_tokens)
                loss = loss_fn(pred, survival_time.unsqueeze(1), event_indicator.unsqueeze(1), device)

                all_preds.append(torch.squeeze(pred).cpu().item()) 
                all_survival_times.append(survival_time.cpu().item())
                all_event_inds.append(event_indicator.cpu().item())
                running_loss += loss.item() * survival_time.size(0)

        else: 
            raise Exception(f"Modality {modality} not among given options")

    avg_loss = running_loss / len(dataloader.dataset) 

    return avg_loss, np.array(all_preds), np.array(all_survival_times), np.array(all_event_inds)



def c_index(risk_pred, surv_times, event_ind):

    risk_pred = risk_pred.detach().cpu().numpy()
    surv_times = surv_times.detach().cpu().numpy()
    event_ind = event_ind.detach().cpu().numpy()

    return concordance_index(surv_times, risk_pred, event_ind)


class EarlyStopping:
    def __init__(self, patience=1, delta=0.):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss: 
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.delta): 
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    