from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import PIL
from PIL.Image import Image
from torch import Tensor
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

from .config import args


class DataSetBase(data.Dataset, ABC):
    def __init__(self, image_path: Path):
        self.images = [PIL.Image.open(img_path).convert("RGB") for img_path in image_path.iterdir()]
        self.max_num_sample = len(self.images)

    def __len__(self) -> int:
        return self.max_num_sample

    @abstractmethod
    def get_low_resolution_image(self, image: Image, path: Path) -> Image:
        pass

    def preprocess_high_resolution_image(self, image: Image) -> Tensor:
        return transforms.ToTensor()(image).to('cuda')

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        high_resolution_image = self.preprocess_high_resolution_image(self.images[index % len(self.images)])
        low_resolution_image = self.get_low_resolution_image(high_resolution_image.cpu(), None)
        return low_resolution_image.to('cuda'), high_resolution_image


class TrainDataSet(DataSetBase):
    def __init__(self, image_path: Path, num_image_per_epoch: int = 2000):
        super().__init__(image_path)
        self.max_num_sample = num_image_per_epoch

    def get_low_resolution_image(self, image: Tensor, path: Path) -> Tensor:
        return transforms.Resize((image.shape[1] // 4, image.shape[2] // 4), transforms.InterpolationMode.BICUBIC)(
            image
        )

    def preprocess_high_resolution_image(self, image: Image) -> Tensor:
        return transforms.Compose(
            [
                transforms.RandomCrop(size=512),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )(image).to('cuda')


class ValidationDataSet(DataSetBase):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path):
        super().__init__(high_resolution_image_path)
        self.low_resolution_images = [
            PIL.Image.open(img_path).convert("RGB") for img_path in low_resolution_image_path.iterdir()
        ]

    def get_low_resolution_image(self, image: Image, path: Path) -> Tensor:
        index = self.images.index(image)
        return transforms.ToTensor()(self.low_resolution_images[index]).to('cuda')


def get_dataset() -> Tuple[DataLoader, DataLoader]:
    train_dataset = TrainDataSet(Path(args.train_data_path), 850 * 10)
    val_dataset = ValidationDataSet(Path(args.val_ori_data_path), Path(args.val_025_data_path))

    return train_dataset, val_dataset
