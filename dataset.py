import random

from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform
import utils
from torch.utils.data import Dataset
from PIL import Image


class ImageFilesDataset(Dataset):

    def __init__(self, image_paths, grayscale=False, training=False):
        super().__init__()

        assert len(image_paths) > 0
        random.shuffle(image_paths)
        self.image_paths = image_paths

        self.transform = SimCLRTrainDataTransform(input_height=32)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        images = self.transform(image)
        return images, 1

    def __len__(self):
        return len(self.image_paths)


class ImageFolderDataset(ImageFilesDataset):

    def __init__(self, image_dir, grayscale=False, training=False):
        image_paths = utils.recursive_folder_image_paths(image_dir)
        super().__init__(image_paths, grayscale, training)
