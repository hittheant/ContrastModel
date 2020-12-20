from argparse import ArgumentParser

import torchvision
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset, DataLoader
from pl_bolts.models.self_supervised import AMDIM
import pytorch_lightning as pl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--save_dir', required=True, type=str, help='path to save resulting edge maps')
    parser.add_argument('--saved_model_path', default='bsds500.pth', type=str, help='path of saved PyTorch model')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    model = AMDIM(encoder='resnet18', pretrained='imagenet2012')

    imagenet_data_train = torchvision.datasets.ImageNet(args.data_dir, split='train')
    imagenet_data_val = torchvision.datasets.ImageNet(args.data_dir, split='val')

    train_data_loader = torch.utils.data.DataLoader(imagenet_data_train,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=2)
    val_data_loader = torch.utils.data.DataLoader(imagenet_data_val,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  num_workers=2)

    trainer = pl.Trainer()
    trainer.fit(model, train_data_loader, val_data_loader)