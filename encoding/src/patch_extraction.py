# Standard Library Imports
import os
from pathlib import Path

# Third-Party Imports
import pyvips
from tqdm import tqdm

# Local Imports
from utils import (
    pad_img,
    get_args, 
    patchify,
    save_patches,
    save_patch_args, 
    get_target_size,
    )


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