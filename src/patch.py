# Standard Library Imports
import os
from pathlib import Path
from typing import List, Tuple

# Third-Party Imports
import pyvips
import tifffile
import numpy as np
from tqdm import tqdm
from empatches import EMPatches

# Local Imports
from utils import get_args, save_patch_args

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
        y1, y2, x1, x2 = coordinates

        patch_name = f"patch-{y1}-{y2}-{x1}-{x2}.ome.tif"
        patch_path = os.path.join(output_dir, patch_name)

        tifffile.imwrite(patch_path, patch, ome=True, metadata={'axes': 'YXC'}, compression="deflate")


def main():
    data_dir = os.path.join("..", "data", "images")
    arg_dir = os.path.join("configs", "patch-config.yaml")
    patch_dir = os.path.join("..", "data", "patches")
    
    os.makedirs(patch_dir, exist_ok=True)

    experiments  = os.listdir(patch_dir)
    experiment_num = len(experiments)
    output_dir = os.path.join(patch_dir, f"experiment-{experiment_num}")
    
    args = get_args(arg_dir)
    save_patch_args(output_dir, args)
    image_names = os.listdir(data_dir)
    image_paths = [os.path.join(data_dir, i) for i in image_names]

    for image_path in tqdm(image_paths, desc="Patchifying images"):
        image_name = Path(image_path).stem.split(".")[0]
        wsi = pyvips.Image.new_from_file(image_path, access="sequential").numpy()

        image_patches, coordinates = patchify(wsi, args["patchsize"], args["overlap"])
        save_patches(
            image_patches, 
            coordinates, 
            os.path.join(output_dir, image_name)
        )


if __name__ == "__main__":
    main()