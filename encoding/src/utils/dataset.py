import os
from pathlib import Path
from typing import Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PatchingDataset(Dataset):
    """
    Creates a dataset class to embed patches in batches.

    Parameters
    ----------
    id_dir: str
        The directory containing the patches (this would be the patient ID).

    Returns
    -------
    img: torch.Tensor
        The patch as a Tensor.

    coords:
        The coordinates of the patch containing its coordinates.
    """

    def __init__(self, id_dir: str) -> None:
        self.id_dir = id_dir
        self.patches = os.listdir(id_dir)

    def __len__(self):
        return len(self.patches)
    
    def valid_patch(self, img: Image, threshold: int = 230) -> bool:

        """
        Checks whether a patch is mostly background.
        Returns false if the patch contains 75% or more background pixels.
        """

        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, background_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        white_pixels = np.sum(background_mask == 255)
        total_pixels = img.shape[0] * img.shape[1]

        background_composition = white_pixels / total_pixels

        validity = bool(background_composition < 0.75)

        return validity
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[str]]:
        img_name = self.patches[idx]
        img_path = os.path.join(self.id_dir, img_name)

        img = Image.open(img_path)
        valid_img = self.valid_patch(img)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )

        img = preprocess(img)
        coords = Path(img_name).stem

        return img, coords, valid_img


