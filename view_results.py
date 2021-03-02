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
    args = parser.parse_args()

    image_paths = utils.recursive_folder_image_paths(args.data_dir)

    model = SimCLR.load_from_checkpoint(checkpoint_path=args.model_dir, strict=False)
    model_enc = model.encoder
    model_enc.eval()

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    y = np.empty((len(image_paths), 2048), float)

    for i, p in enumerate(tqdm(image_paths)):
        image = Image.open(p)
        image = image.convert('RGB')
        image = transform(image).unsqueeze_(0)
        y_hat = model_enc(image)
        y_hat = y_hat[0].detach().numpy().reshape(1, -1)
        y[i, :] = y_hat

    print(y)
    y_comp = np.abs(np.array(y) - np.array(y[32]))
    y_comp = np.sum(y_comp, axis=1)
    sort_index = np.argsort(y_comp)
    print(sort_index[0:20])
    for index in sort_index[0:20]:
        image = Image.open(image_paths[index])
        image.save('results/' + str(index) + '.jpeg')
