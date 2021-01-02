import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageFilesDataset(Dataset):

    def __init__(self, image_paths, grayscale=False, training=False):
        super().__init__()

        assert len(image_paths) > 0
        random.shuffle(image_paths)
        self.image_paths = image_paths

        transform = [transforms.Resize((64, 64))]
        if training:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform)

        self.nc = 1 if grayscale else 3

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = image.convert('L') if self.nc == 1 else image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)