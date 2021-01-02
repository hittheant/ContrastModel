import utils
import torch
import os
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pl_bolts.models.self_supervised import AMDIM
from argparse import ArgumentParser
from dataset import ImageFilesDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--save_dir', required=True, type=str, help='path to save resulting edge maps')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    model = AMDIM(encoder='resnet18', pretrained='imagenet2012')

    files = utils.recursive_folder_image_paths(args.data_dir)
    random.shuffle(files)
    train_files = files[:int(args.train_test_ratio * len(files))]
    test_files = files[int(args.train_test_ratio * len(files)):]
    train_set = ImageFilesDataset(train_files, grayscale=args.grayscale, training=True)
    test_set = ImageFilesDataset(test_files, grayscale=args.grayscale, training=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    trainer = pl.Trainer()
    trainer.fit(model, train_loader, test_loader)
