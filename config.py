import argparse
from dataclasses import dataclass



@dataclass
class TrainingConfig:
    modality: str
    epochs: int
    fold_value: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    freeze_option: str
    deepsurv_hidden_layers: list[int]
    deepsurv_dropout_prob: float
    early_stopping_patience: int
    images_dir: str
    tabular_data_dir: str
    pretrained_weights_dir: str
    save_dir: str
    cuda_choice: str
    seed: int


def parse_args() -> TrainingConfig:

    parser = argparse.ArgumentParser(description='Train CT-CLIP survival models')

    # MODEL
    parser.add_argument('--modality', type=str, default='multimodal', choices=['multimodal', 'image', 'text'], help='Modality to use (default: multimodal)') 
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--foldval', type=int, default=0, choices=[0,1,2], help='Cross validation fold (default: 0)')
    parser.add_argument('--batchsize', type=int, default=4, help='Training batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
    parser.add_argument('--wd', type=float,default=1e-5, help='Weight decay (default: 1e-5)')
    parser.add_argument('--freezeoption', type=str, default='trds', choices=['trall', 'trvbds', 'trvds', 'trvvpds', 'trvvptpds', 'trbds', 'trbtpds', 'trbvptpds', 'trvptpds', 'trds', 'trvpds', 'trtpds'], help='Freeze option (default: trds)') 
    parser.add_argument('--hiddenlayers', type=int, nargs="+", default=[512, 256], help='Number of neurons in each hidden layer in DeepSurv (default: [512, 256])')
    parser.add_argument('--dropoutprob', type=float, default=0.25, help='Dropout probability in DeepSurv (default: 0.25)') 
    parser.add_argument('--earlystop', type=int, default=10, help='Early stopping patience (default: 10)') 

    # DIRECTORIES
    parser.add_argument('--imagedir', type=str, default="/preprocessed_images", help='Path to the directory containing preprocessed images')
    parser.add_argument('--tabulardir', type=str, default="/tabular_data.csv", help='Path to the CSV file containing patient outcome data, clinical data, and fold assignments.')
    parser.add_argument('--ctclipdir', type=str, default="/CT-CLIP_v2.pt", help='Path to pretrained CT-CLIP weights')
    parser.add_argument('--savedir', type=str, default="/save_dir", help='Where to save the trained model')

    parser.add_argument('--cuda', type=str, default='cuda:0', choices=['cuda:0','cuda:1'], help='Choice of GPU')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability (default: 0)')

    args = parser.parse_args()

    config = TrainingConfig(
        modality=args.modality, 
        epochs=args.epochs,
        fold_value=args.foldval,
        batch_size=args.batchsize,
        learning_rate=args.lr,
        weight_decay=args.wd,
        freeze_option=args.freezeoption,
        deepsurv_hidden_layers=args.hiddenlayers,
        deepsurv_dropout_prob=args.dropoutprob,
        early_stopping_patience=args.earlystop,
        images_dir = args.imagedir,
        tabular_data_dir = args.tabulardir,
        pretrained_weights_dir = args.ctclipdir,
        save_dir = args.savedir,
        cuda_choice=args.cuda,
        seed=args.seed
    )

    return config



