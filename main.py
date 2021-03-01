import torch
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pl_bolts.models.self_supervised import SimCLR
from argparse import ArgumentParser
from dataset import ImageFolderDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--save_dir', required=True, type=str, help='path to save resulting edge maps')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    train_set = ImageFolderDataset(os.path.join(args.data_dir, 'train'), training=True)
    test_set = ImageFolderDataset(os.path.join(args.data_dir, 'test'), training=False)

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=12, shuffle=False, num_workers=4)

    model = SimCLR(gpus=1, num_samples=(len(train_set) + len(test_set)),
                   batch_size=12, dataset=train_loader)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, test_loader)
    model.freeze()
