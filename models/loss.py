
import torch
import torch.nn as nn

class PartialLogLikelihood(nn.Module):
    def __init__(self):
        super(PartialLogLikelihood, self).__init__()

    def forward(self, risk_preds, surv_times, event_inds, device):

        mask = torch.ones(surv_times.shape[0], surv_times.shape[0]).to(device) 
        mask[(torch.transpose(surv_times, 0, 1) - surv_times) > 0] = 0  
        log_loss = torch.exp(risk_preds) * mask
        log_loss = torch.sum(log_loss, dim=0) 
        log_loss = torch.log(log_loss).reshape(-1, 1)

        neg_log_loss = -torch.sum((risk_preds-log_loss) * event_inds) / (torch.sum(event_inds) + 1e-5)

        return neg_log_loss
    
