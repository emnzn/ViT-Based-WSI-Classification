import os
from typing import List

import timm
import torch
import pyvips
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader

from utils import *


def embed_patches(
    model: timm.models.vision_transformer.VisionTransformer,
    dataloader: PatchingDataset,
    device: str,
    save_dir: str
    ) -> torch.Tensor:

    results = {
        "coords":[],
        "embedding": []
    }

    model.eval()

    with torch.inference_mode():
        for img, coords in tqdm(dataloader, desc="Embedding Patches"):
            img = img.to(device)
            embedding = model(img).cpu().numpy()

            results["coords"].extend(coords)
            results["embedding"].extend(embedding)

    save_embeddings(results, save_dir)


def main():
    arg_dir = os.path.join("configs", "embed-config.yaml")
    args = get_args(arg_dir)
    
    data_dir = os.path.join("..", "..", "data", "patches", f"experiment-{args['experiment_num']}")
    dest_dir = os.path.join("..", "..", "data", "embeddings", f"experiment-{args['experiment_num']}")

    os.makedirs(dest_dir, exist_ok=True)
    patient_ids = [id for id in os.listdir(data_dir) if "." not in id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args["model"], device)

    for i, id in enumerate(patient_ids):
        print(f"patient: {id} | [{i+1}/{len(patient_ids)+1}]")

        id_dir = os.path.join(data_dir, id)
        save_dir = os.path.join(dest_dir, f"{id}.parquet")

        patch_dataset = PatchingDataset(id_dir)
        patch_loader = DataLoader(patch_dataset, batch_size=args["batch_size"], shuffle=False)
        embed_patches(model, patch_loader, device, save_dir)

        print(f"\nembeddings saved")
        print("-------------------------------------------------------------------\n")
        break


if __name__ == "__main__":
    main()