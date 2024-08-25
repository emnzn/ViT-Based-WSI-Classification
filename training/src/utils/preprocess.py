from pathlib import Path
from typing import Tuple, List

import numpy as np

def extract_coords(img_name: str) -> Tuple[int, int, int, int]:

    """
    Extracts the coordinates of a given patch from its filename.
    """

    stem = Path(img_name).stem
    y1, y2, x1, x2 = [int(c) for c in stem.split("-")[1:]]
    coords = (y1, y2, x1, x2)

    return coords


def merge_patches(
    patches: List[np.ndarray], 
    coords: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:

    """
    Merges patches into the whole image given a list of coordinates.

    Parameters
    ----------
    patches: List[np.ndarray]
        A list containing the patches.

    coords: List[Tuple[int]]
        A list of coordinates specifying the location of each patch in the entire image.

    Returns
    -------
    merged_img: np.ndarray
        The merged image.
    """
    
    orig_height = max([c[1] for c in coords])
    orig_width = max([c[3] for c in coords])
    
    num_channels = patches[0].shape[-1]
    merged_img = np.zeros((orig_height, orig_width, num_channels), dtype=patches[0].dtype)

    for i, coord in enumerate(coords):
        merged_img[coord[0] : coord[1], coord[2] : coord[3], :] = patches[i]

    return merged_img


def get_checkpoint(coords: List[Tuple[int]]) -> int:

    """
    Given a list of coordinates, gets the checkpoint.

    The coordinates are in the form (y1, y2, x1, x2),
    and must be sorted from left to right, then top to bottom.

    The checkpoint is the index of the first coordinate where the x1 value 
    changes from 0 to another value, indicating a shift to the next column.
    Suppose given a (448, 448) image wherein (224, 224) patches have been created:

    Example:
    [
        (0, 224, 0, 224),
        (224, 448, 0, 224),
        (0, 224, 224, 448),  -> this is the checkpoint (index 2)
        (224, 448, 224, 448)
    ]

    Returns
    -------
    checkpoint: int 
        The index of the first coordinate where x1 changes.
    """

    for i, index in enumerate(coords):
        if index[2] != 0:
            checkpoint = i
            
            return checkpoint


def adjust_coords(
    coords: List[Tuple[int, int, int, int]], 
    new_size: int
    ) -> List[Tuple[int, int, int, int]]:

    """
    To stitch embeddings back to the original shape, coordinates need to be adjusted according to the new dimensions.

    Given a set of coordinates (sorted according to x1, x2, y1, y2), this scripts makes that adjustment

    Parameters
    ----------
    coords: List[Tuple[int]]
        A list of coordinates sorted accoring to (x1, x2, y1, y2)

    new_size: int
        The new dimensions of each patch.
        This can only cater to dimensions where height and width are equal.

    Returns
    -------
    adjusted_coords: List[Tuple[int]]
        The new coordinates of each patch adjusted acording to its new dimensions.
    """

    checkpoint = get_checkpoint(coords)
    placeholder = [0, new_size, 0, new_size]
    adjusted_coords = []

    for i in range(len(coords)):
        if i % checkpoint == 0 and i > 0:
            placeholder[0] = 0
            placeholder[1] = new_size

            placeholder[2] += new_size
            placeholder[3] += new_size

        y1, y2, = placeholder[0], placeholder[1]
        x1, x2 = placeholder[2], placeholder[3]
        adjusted_coords.append((y1, y2, x1, x2))

        placeholder[0] += new_size
        placeholder[1] += new_size

    return adjusted_coords