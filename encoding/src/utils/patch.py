import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from empatches import EMPatches


def patchify(
    wsi: np.ndarray,
    patchsize: int,
    overlap: float
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:

    """
    Patchify a whole slide image (WSI) into smaller patches.

    Parameters
    ----------
    wsi: np.ndarray
        The whole slide image as a numpy array.

    patchsize: int
        The size of each patch.

    overlap: float
        The overlap between patches.

    Returns
    -------
    image_patches: List[np.ndarray]
        A list of image patches.

    coordinates: List[Tuple[int, int, int, int]]
        A list of tuples containing the coordinates of each patch.
    """

    emp = EMPatches()
    image_patches, coordinates = emp.extract_patches(wsi, patchsize=patchsize, overlap=overlap)

    return image_patches, coordinates


def save_patches(
    image_patches: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    output_dir: str,
    ) -> None:

    """
    Save the image patches as individual images.

    Parameters
    ----------
    image_patches: List[np.ndarray]
        A list of image patches.

    indices: List[Tuple[int, int, int, int]]
        A list of tuples containing the coordinates of each patch.

    output_dir: str
        The directory where the image patches will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    for patch, coordinates in zip(image_patches, coordinates):
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        y1, y2, x1, x2 = coordinates

        patch_name = f"patch-{y1}-{y2}-{x1}-{x2}.png"
        patch_path = os.path.join(output_dir, patch_name)

        cv2.imwrite(patch_path, patch)


def extract_coords(img_name: List[str]) -> Tuple[int]:

    """
    Extracts the coordinates of a given patch from its filename.
    """

    stem = Path(img_name).stem
    y1, y2, x1, x2 = [int(c) for c in stem.split("-")[1:]]
    coords = (y1, y2, x1, x2)

    return coords


def get_background_color(
    image: np.ndarray,
    threshold: int = 230
    ) -> List[int]:

    """
    Gets the average background color of the image.

    Parameters
    ----------
    image: np.array
        The whole slide image in the form of a numpy array.

    threshold: int
        The threshold for binary conversion. (default: 230)

    Returns
    -------
    background_color: List[int]
        The average background color of the whole slide image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    background_intensity = np.mean(image[mask == 255])
    background_color = (int(background_intensity),) * 3

    return background_color


def get_nearest_multiple(source, target) -> int:

    """
    Calculate the nearest multiple of a target number to a given source number.

    Parameters
    ----------
    source : int
        The number from which to find the nearest multiple.

    target : int
        The number whose multiple is to be found.

    Returns
    -------
    nearest_multiple: int
        The nearest multiple of `target` to `source`. If `source` is already a multiple of `target`, it returns `source`.
    
    Notes
    -----
    This function rounds up to the next multiple of `target` if `source` is not already a multiple of `target`.
    """
    
    remainder = source % target

    nearest_multiple = (source + (target - remainder)) if remainder else source
    
    return nearest_multiple


def get_target_shape(img: np.ndarray, target: int) -> Tuple[int]:

    """
    Calculates the target shape of an image, to become a multiple of a target number.

    Parameters
    ----------
    img: np.ndarray
        The image.

    target: int
        The target number.

    Returns
    -------
    target_size: Tuple[int]
        The size the image should be to become a multiple of the target number.
    """

    source_height, source_width = img.shape[0], img.shape[1]

    target_height = get_nearest_multiple(source_height, target)
    target_width = get_nearest_multiple(source_width, target)

    target_size = (target_height, target_width)

    return target_size


def pad_img(img: np.ndarray, target_shape: Tuple[int]) -> np.ndarray:

    """
    Pads the image to a target shape.
    """

    bg_color = get_background_color(img)

    current_shape = img.shape[:2]

    delta_h = target_shape[0] - current_shape[0]
    delta_w = target_shape[1] - current_shape[1]

    h_pad = delta_h // 2
    w_pad = delta_w // 2

    padded_img = cv2.copyMakeBorder(img, h_pad, h_pad, w_pad, w_pad, borderType=cv2.BORDER_CONSTANT, value=bg_color)

    return padded_img