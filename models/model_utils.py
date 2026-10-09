import torch
from peft import LoraConfig
from transformers import BertTokenizer, BertModel
from models.ct_vit import CTViT
from models.nn_models import CTClipDeepSurv, BERTDeepSurv, CTViTDeepSurv



def build_model(modality, ds_hidden_layers, ds_dropout_prob, device):

    if modality == "multimodal":

        print("\nMODEL:\n------\nUsing model CTClipDeepSurv")

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            device=device,
            dim_head=32,
            heads=8
        )

        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(tokenizer))

        model = CTClipDeepSurv(
            image_encoder,
            text_encoder, 
            dim_latent=512,
            ds_hidden_layers=ds_hidden_layers, 
            ds_dropout_prob=ds_dropout_prob
            )

    elif modality == "image":

        print("\nMODEL:\n------\nUsing model CTViTDeepSurv")

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            device=device, 
            dim_head=32,
            heads=8
        )

        model = CTViTDeepSurv(
            image_encoder,
            dim_latent=512,
            ds_hidden_layers=ds_hidden_layers, 
            ds_dropout_prob=ds_dropout_prob
            )

        tokenizer = None

    elif modality == "text":

        print("\nMODEL:\n------\nUsing model BERTDeepSurv")

        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(tokenizer))

        model = BERTDeepSurv(
            text_encoder=text_encoder,
            dim_text=768,
            dim_latent=512,
            ds_hidden_layers=ds_hidden_layers, 
            ds_dropout_prob=ds_dropout_prob
            )

    else: 
        raise Exception(f"Modality {modality} not among given options")

    return model, tokenizer



def load_pretrained_backbone(model, modality, weights_dir):
    if modality == "multimodal":
        load_pretrained_weights_CTClip(model, weights_dir)
    elif modality == "image":
        load_pretrained_weights_CTViT(model, weights_dir)
    elif modality == "text":
        load_pretrained_weights_BERT(model, weights_dir)
    else: 
        raise Exception(f"Modality {modality} not among given options")


def load_pretrained_weights_CTClip(model, weights_dir):
    pretrained = torch.load(weights_dir, weights_only=True, map_location="cpu") 
    result = model.load_state_dict(pretrained, strict=False)

    missing_keys = [item for item in result.missing_keys if not "deepsurv" in item]
    if len(missing_keys) > 1:
        print("\nWeights in vision and text transformer and projection layers were not correctly updated. Loading might have failed.")
    else:
        print("\nWeights in vision and text_transformer and projection layers were successfully loaded.")


def load_pretrained_weights_CTViT(model, weights_dir):
    pretrained = torch.load(weights_dir, weights_only=True, map_location="cpu") 
    vision_transformer_weights = {k: v for k, v in pretrained.items() if "visual_transformer" in k or "to_visual_latent.weight" in k} 
    result = model.load_state_dict(vision_transformer_weights, strict=False) 

    missing_keys = [item for item in result.missing_keys if not "deepsurv" in item]
    if len(missing_keys) > 1:
        print("\nWeights in vision transformer and projection layer were not correctly updated. Loading might have failed.")
    else:
        print("\nWeights in vision_transformer and projection layer were successfully loaded.\n")


def load_pretrained_weights_BERT(model, weights_dir):
    pretrained = torch.load(weights_dir, map_location="cpu") 
    text_transformer_weights = {k: v for k, v in pretrained.items() if "text_transformer" in k or "to_text_latent.weight" in k} 
    result = model.load_state_dict(text_transformer_weights, strict=False) 

    missing_keys = [item for item in result.missing_keys if not "deepsurv" in item]
    if len(missing_keys) > 0: 
        print("\nWeights in text transformer and projection layer were not correctly updated. Loading might have failed.")
    else:
        print("\nWeights in text_transformer and projection layer were successfully loaded.\n")


def load_model_weights(model, weights_dir):
    model_dict = torch.load(weights_dir, map_location="cpu")
    model_weights = model_dict['model_state_dict']
    model.load_state_dict(model_weights, strict=True)



def freeze_layers_based_on_freeze_option(model, freeze_option, modality):
    
    if modality == "multimodal":
        freeze_layers_based_on_freeze_option_multimodal(model, freeze_option, modality)
    elif modality == "image":
        freeze_layers_based_on_freeze_option_multimodal_image(model, freeze_option, modality)
    elif modality == "text":
        freeze_layers_based_on_freeze_option_multimodal_text(model, freeze_option, modality)
    else: 
        raise Exception(f"Selected modality {modality} not among given options")


def freeze_layers_based_on_freeze_option_multimodal(model, freeze_option, modality):
    """
    Available freezing options multimodal model: 'trall', 'trvbds', 'trvds', 'trvvpds', 'trvvptpds', 
        'trbds', 'trbtpds', 'trbvptpds', 'trvptpds', 'trds'

        'trall' - train all parts
        'trvbds' - train ViT, BERT, DS
        'trvds' - train ViT, DS
        'trvvpds' - train ViT, VP, DS
        'trvvptpds' - train ViT, VP, TP, DS
        'trbds' - train BERT, DS
        'trbtpds' - train BERT, TP, DS
        'trbvptpds' - train BERT, VP, TP, DS
        'trvptpds' - train VP, TP, DS
        'trds' - train DS
    """

    if freeze_option == "trall":
        pass
    elif freeze_option == "trvbds":
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trvds": 
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
        for param in model.text_transformer.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trvvpds":
        for param in model.text_transformer.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trvvptpds": 
        for param in model.text_transformer.parameters():
            param.requires_grad = False
    elif freeze_option == "trbds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trbtpds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trbvptpds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
    elif freeze_option == "trvptpds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
        for param in model.text_transformer.parameters():
            param.requires_grad = False
    elif freeze_option == "trds":
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
        for param in model.text_transformer.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    else: 
        raise Exception(f"Chosen freeze options not among given options for modality: {modality}.")


def freeze_layers_based_on_freeze_option_multimodal_image(model, freeze_option, modality):
    """
    Available freezing options image model: 'trvds', 'trvvpds', 'trvpds', 'trds'

        'trvds' - train ViT, DS
        'trvvpds' - train ViT, VP, DS
        'trvpds' - train VP, DS
        'trds' - train DS
    """
    if freeze_option == "trvds":
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trvvpds": 
        pass 
    elif freeze_option == "trvpds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
    elif freeze_option == "trds": 
        for param in model.visual_transformer.parameters():
            param.requires_grad = False
        for param in model.to_visual_latent.parameters():
            param.requires_grad = False
    else: 
        raise Exception(f"Chosen freeze options not among given options for modality: {modality}.")


def freeze_layers_based_on_freeze_option_multimodal_text(model, freeze_option, modality):
    """
    Available freezing options text model: 'trbds', 'trbtpds', 'trtpds', 'trds'

        'trbds' - train BERT, DS
        'trbtpds' - train BERT, TP, DS
        'trtpds' - train TP, DS
        'trds' - train DS
    """

    if freeze_option == "trbds": 
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    elif freeze_option == "trbtpds": 
        pass
    elif freeze_option == "trtpds": 
        for param in model.text_transformer.parameters():
            param.requires_grad = False
    elif freeze_option == "trds": 
        for param in model.text_transformer.parameters():
            param.requires_grad = False
        for param in model.to_text_latent.parameters():
            param.requires_grad = False
    else: 
        raise Exception(f"Chosen freeze options not among given options for modality: {modality}.")


def get_lora_target_modules_from_freeze_option(freeze_option, modality):
    """
    Available options multimodal model: 'trall', 'trvbds', 'trvds', 'trvvpds', 'trvvptpds', 
        'trbds', 'trbtpds', 'trbvptpds', 'trvptpds'

        'trall' - train all parts
        'trvbds' - train ViT, BERT, DS
        'trvds' - train ViT, DS
        'trvvpds' - train ViT, VP, DS
        'trvvptpds' - train ViT, VP, TP, DS
        'trbds' - train BERT, DS
        'trbtpds' - train BERT, TP, DS
        'trbvptpds' - train BERT, VP, TP, DS
        'trvptpds' - train VP, TP, DS

    Available options image model: 'trvds', 'trvvpds', 'trvpds'

        'trvds' - train ViT, DS
        'trvvpds' - train ViT, VP, DS
        'trvpds' - train VP, DS

    Available options text model: 'trbds', 'trbtpds', 'trtpds'

        'trbds' - train BERT, DS
        'trbtpds' - train BERT, TP, DS
        'trtpds' - train TP, DS
    
    """

    if modality == "multimodal":

        if freeze_option == "trvbds":
            lora_target_modules = ["query", "value", "to_q", "to_kv"]
        elif freeze_option == "trbds":
            lora_target_modules = ["query", "value"]
        elif freeze_option == "trvds":
            lora_target_modules = ["to_q", "to_kv"]
        elif freeze_option == "trall":
            lora_target_modules = ["query", "value", "to_q", "to_kv", "to_text_latent", "to_visual_latent"]
        elif freeze_option == "trvvpds":
            lora_target_modules = ["to_q", "to_kv", "to_visual_latent"]
        elif freeze_option == "trvvptpds":
            lora_target_modules = ["to_q", "to_kv", "to_text_latent", "to_visual_latent"]
        elif freeze_option == "trbtpds":
            lora_target_modules = ["query", "value", "to_text_latent"]
        elif freeze_option == "trbvptpds":
            lora_target_modules = ["query", "value", "to_text_latent", "to_visual_latent"]
        elif freeze_option == "trvptpds":
            lora_target_modules = ["to_text_latent", "to_visual_latent"]
        else: 
            raise Exception(f"Selected freeze option not implemented for LoRA finetuning with modality {modality}")
        
    elif modality == "image":

        if freeze_option == "trvds":
            lora_target_modules = ["to_q", "to_kv"]
        elif freeze_option == "trvvpds":
            lora_target_modules = ["to_q", "to_kv", "to_visual_latent"]
        elif freeze_option == "trvpds":
            lora_target_modules = ["to_visual_latent"]
        else: 
            raise Exception(f"Selected freeze option not implemented for LoRA finetuning with modality {modality}")

    elif modality == "text":

        if freeze_option == "trbds":
            lora_target_modules = ["query", "value"]
        elif freeze_option == "trbtpds":
            lora_target_modules = ["query", "value", "to_text_latent"]
        elif freeze_option == "trtpds":
            lora_target_modules = ["to_text_latent"]
        else: 
            raise Exception(f"Selected freeze option not implemented for LoRA finetuning with modality {modality}")

    else: 
        raise Exception(f"Selected modality {modality} not among given options")
    
    return lora_target_modules


def get_lora_config(freeze_option, modality):

    lora_target_modules = get_lora_target_modules_from_freeze_option(freeze_option, modality)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=lora_target_modules, 
        bias="none", 
        lora_dropout=0.05, 
        modules_to_save=["deepsurv"]
    )
    return lora_config


def print_trainable_layers(model):
    print("Trainable parameters (requires_grad=True):")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
