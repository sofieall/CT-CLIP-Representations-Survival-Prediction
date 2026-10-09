# CT-CLIP Representations for Multimodal Lung Cancer Survival Prediction

### Training

#### Preprocessing of CT images
The images loaded by the dataset class have already been preprocessed. To preprocess the images and save them to disk, run `preprocess_and_save_images.py`.

#### Using the main training scripts
```bash
# Train multimodal model with frozen CT-CLIP backbone
python train.py --modality "multimodal" --epochs 100 --foldval 0 --freezeoption "trds"  

# Train multimodal model with full fine-tuning of ViT and BERT
python train.py --modality "multimodal" --epochs 100 --foldval 0 --freezeoption "trvbds"

# Train multimodal model with Low-Rank Adaptation
python train_lora.py --modality "multimodal" --epochs 100 --foldval 0 --freezeoption "trvbds"
```

## Dependencies
The project was developed using Python 3.11 and the packages listed in `requirements.txt`. The development environment used PyTorch 2.5.1 with CUDA 12.1 on an NVIDIA GPU. This is provided for reference and is not intended to require an identical environment.

## Acknowledgments and License
Some scripts in this repository are based on code from:

**CT-CLIP**  
Source: https://github.com/ibrahimethemhamamci/CT-CLIP  
License: [Creative Commons Attribution (CC-BY-NC-SA) license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
