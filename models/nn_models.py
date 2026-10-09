
import torch.nn as nn
import torch.nn.functional as F
import torch


def l2norm(t):
    return F.normalize(t, dim = -1)


class BERTDeepSurv(nn.Module):

    def __init__(self, text_encoder, dim_text=768, dim_latent=512, ds_hidden_layers=[512, 256], ds_dropout_prob=0.25):  

        super().__init__()

        self.text_transformer = text_encoder
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)
        self.deepsurv = DeepSurv(input_dim=dim_latent, 
                                 hidden_layers=ds_hidden_layers, 
                                 dropout_prob=ds_dropout_prob)

    def forward(self, text_tokens):

        text_embeddings = self.text_transformer(text_tokens.input_ids, attention_mask=text_tokens.attention_mask)
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
 
        text_embeds = text_embeds[:,0,:]
        text_latents = self.to_text_latent(text_embeds)
        text_latents = l2norm(text_latents)

        logits = self.deepsurv(text_latents)

        return logits
  


class CTViTDeepSurv(nn.Module):

    def __init__(self, image_encoder, dim_latent=512, ds_hidden_layers=[512, 256], ds_dropout_prob=0.25): 

        super().__init__()

        self.visual_transformer = image_encoder
        dim_image = 294912 # from init CT-CLIP
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)
        self.deepsurv = DeepSurv(input_dim=dim_latent,
                                 hidden_layers=ds_hidden_layers, 
                                 dropout_prob=ds_dropout_prob)

    def forward(self, preprocessed_image):

        enc_image = self.visual_transformer(preprocessed_image, return_encoded_tokens=True)
        if len(enc_image.shape) != 5:
            raise Exception("Image is not encoded with original CT-ViT")
        
        enc_image = torch.mean(enc_image, dim=1)
        enc_image = enc_image.view(enc_image.shape[0], -1)
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        image_latents = self.to_visual_latent(image_embeds)
        image_latents = l2norm(image_latents)
        logits = self.deepsurv(image_latents)

        return logits



class CTClipDeepSurv(nn.Module):

    def __init__(self, image_encoder, text_encoder, dim_latent=512, ds_hidden_layers=[512, 256], ds_dropout_prob=0.25): 

        super().__init__()

        self.visual_transformer = image_encoder
        dim_image = 294912 # from init CT-CLIP
        self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        self.text_transformer = text_encoder
        dim_text=768 # from init CT-CLIP
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        self.deepsurv = DeepSurv(input_dim=dim_latent*2,
                                 hidden_layers=ds_hidden_layers, 
                                 dropout_prob=ds_dropout_prob)  


    def forward(self, preprocessed_image, text_tokens):

        # Image
        enc_image = self.visual_transformer(preprocessed_image, return_encoded_tokens=True)
        if len(enc_image.shape) != 5:
            raise Exception("Image is not encoded with original CT-ViT")
        enc_image = torch.mean(enc_image, dim=1)
        enc_image = enc_image.view(enc_image.shape[0], -1)
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
        image_latents = self.to_visual_latent(image_embeds)

        # Text
        text_embeddings = self.text_transformer(text_tokens.input_ids, attention_mask=text_tokens.attention_mask)
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text
        text_embeds = text_embeds[:,0,:]
        text_latents = self.to_text_latent(text_embeds)

        text_latents, image_latents = map(l2norm, (text_latents, image_latents))

        # Concatenation
        concatenated_latents = torch.cat((image_latents, text_latents), 1)

        logits = self.deepsurv(concatenated_latents)

        return logits



class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_prob=0.5):
        super(DeepSurv, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)