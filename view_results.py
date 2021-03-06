import utils
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from pl_bolts.models.self_supervised import SimCLR
from argparse import ArgumentParser
from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser(description='Get image edge maps')
    parser.add_argument('--model_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--data_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--save_dir', required=True, type=str, help='path to image data directory')
    parser.add_argument('--image_index', type=int, default=42)
    parser.add_argument('--n_images', type=int, default=20)
    parser.add_argument('--rgb', type=bool, default=True)
    args = parser.parse_args()

    image_paths = utils.recursive_folder_image_paths(args.data_dir)

    model = SimCLR.load_from_checkpoint(checkpoint_path=args.model_dir, strict=False)
    model_enc = model.encoder
    model_enc.eval()

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    y = np.empty((len(image_paths), 2048), float)

    for i, p in enumerate(tqdm(image_paths)):
        image = Image.open(p)
        if args.rgb:
            image = image.convert('RGB')
        image = transform(image).unsqueeze_(0)
        y_hat = model_enc(image)
        y_hat = y_hat[0].detach().numpy().reshape(1, -1)
        y[i, :] = y_hat

    print(y)
    y_comp = np.abs(np.array(y) - np.array(y[args.image_index]))
    y_comp = np.sum(y_comp, axis=1)
    sort_index = np.argsort(y_comp)
    print(sort_index[0:args.n_images])
    for i, index in enumerate(sort_index[0:args.n_images]):
        image = Image.open(image_paths[index])
        image.save(args.save_dir + str(i) + "_" + str(index) + '.jpeg')
