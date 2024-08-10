from pathlib import Path
from typing import Tuple

import numpy as np
import PIL
from PIL import Image
from torch import Tensor
from torch.utils import data
from torchvision import transforms

from .config import args


class CachedDataSet(data.Dataset):
    def __init__(self, high_resolution_image_path: Path, low_resolution_image_path: Path, cache_dir: Path):
        self.high_resolution_image_path = high_resolution_image_path
        self.low_resolution_image_path = low_resolution_image_path
        self.cache_dir = cache_dir
        self.images = list(high_resolution_image_path.iterdir())
        self.cache = {}

        # Convert images to .npy format and save in cache_dir if not already converted
        self._prepare_cache()

    def _prepare_cache(self):
        for image_path in self.images:
            cache_path = self.cache_dir / (image_path.stem + '.npy')
            if not cache_path.exists():
                high_res_image = np.array(PIL.Image.open(image_path))
                low_res_image = np.array(
                    PIL.Image.open(
                        self.low_resolution_image_path / image_path.relative_to(self.high_resolution_image_path)
                    )
                )
                np.save(cache_path, {'high': high_res_image, 'low': low_res_image})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if index not in self.cache:
            image_path = self.images[index % len(self.images)]
            cache_path = self.cache_dir / (image_path.stem + '.npy')
            data = np.load(cache_path, allow_pickle=True).item()

            high_resolution_image = PIL.Image.fromarray(data['high'])
            low_resolution_image = PIL.Image.fromarray(data['low'])

            high_resolution_image = self.preprocess_high_resolution_image(high_resolution_image)
            low_resolution_image = self.preprocess_low_resolution_image(low_resolution_image)

            self.cache[index] = (
                transforms.ToTensor()(low_resolution_image),
                transforms.ToTensor()(high_resolution_image),
            )
        return self.cache[index]

    def preprocess_high_resolution_image(self, image: Image) -> Image:
        return transforms.Compose(
            [transforms.RandomCrop(size=512), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )(image)

    def preprocess_low_resolution_image(self, image: Image) -> Image:
        return image


def get_cached_dataset() -> Tuple[CachedDataSet, CachedDataSet]:
    train_cache_dir = Path(args.train_cache_dir)
    val_cache_dir = Path(args.val_cache_dir)
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CachedDataSet(
        high_resolution_image_path=Path(args.train_data_path),
        low_resolution_image_path=Path(args.train_low_res_data_path),
        cache_dir=train_cache_dir,
    )

    validation_dataset = CachedDataSet(
        high_resolution_image_path=Path(args.val_ori_data_path),
        low_resolution_image_path=Path(args.val_025_data_path),
        cache_dir=val_cache_dir,
    )

    return train_dataset, validation_dataset
