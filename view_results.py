import utils
import numpy as np
import torch
import torchvision.transforms as transforms
from pl_bolts.models.self_supervised import SimCLR
import tensorflow as tf
from argparse import ArgumentParser
from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--model_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
    args = parser.parse_args()

    image_paths = utils.recursive_folder_image_paths(args.data_dir)

    model = SimCLR(batch_size=32, num_samples=len(image_paths))
    try:
        model.load_from_checkpoint(args.model_dir)
    except RuntimeError as e:
        print('Ignoring "' + str(e) + '"')

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    model.eval()
    x = torch.empty(len(image_paths), 3, 32, 32)
    for i, p in enumerate(image_paths):
        image = Image.open(p)
        image = image.convert('RGB')
        image = transform(image).unsqueeze_(0)
        x[i, :, :, :] = image
    print(x.shape)
    y_hat = (model(x))
    y_hat = y_hat.detach().numpy().reshape(y_hat.shape[0], -1)
    #y_hat = y_hat / y_hat.max(axis=0)
    print(y_hat.shape)
    y_comp = np.abs(np.array(y_hat) - np.array(y_hat[80]))
    y_comp = np.sum(y_comp, axis=1)
    sort_index = np.argsort(y_comp)
    print(sort_index[0:5])
    for index in sort_index[0:5]:
        image = Image.open(image_paths[index])
        image.save('results/' + str(index) + '.jpeg')
