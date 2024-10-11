import os
from typing import Tuple, Union

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import (
    WSIDataset, get_args, get_save_dirs,
    get_model, ResNet, SwinTransformer, 
    BaseMIL, AttentionBasedMIL,
    multithread_read_img, adjust_coords,
    visualize_attention
)

@torch.no_grad()
def forward_pass(
    dataloader: DataLoader,
    model: Union[ResNet, SwinTransformer, BaseMIL, AttentionBasedMIL],
    patient_table_dir: str,
    patch_dir: str,
    patch_shape: int,
    save_dir: str,
    model_name: str,
    mil: bool,
    device: str,
    ) -> Tuple[float, float, float]:

    """
    Runs inference on a given model.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    model: Union[ResNet, SwinTransformer, AttentionBasedMIL]
        The model to be trained.

    patient_table_dir: str
        The directory with the patient's embedding tables.

    patch_dir: str
        The directory with the patches per WSI.

    patch_shape: int
        The directory with the image patches.

    save_dir: str
        The directory to save the results. 
        The results will be saved as a csv file where the prediction
        for each patient id can be accessed.

    model_name: str
        The name of the model to be saved.

    mil: bool
        Whether training under a Multiple-Instance Learning scheme.
        This is used because the Attenion-Based MIL model returns the 
        attention weights placed on each instance.

    device: str
        One of [cuda, cpu].

    """

    model.eval()
    for wsi_embedding, target, patient_id in dataloader:
        patient_id = patient_id[0]
        embedding_dir = os.path.join(patient_table_dir, f"{patient_id}.parquet")
        patient_table = pd.read_parquet(embedding_dir).drop("embedding", axis=1)
        
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if mil: 
            logits, attention = model(wsi_embedding)

        else: 
            logits = model(wsi_embedding)

        patient_table["adjusted_coords"] = adjust_coords(patient_table["processed_coords"].tolist(), 224, patch_shape)
        patient_table["img_paths"] = patient_table["coords"].map(lambda x: os.path.join(patch_dir, patient_id, f"{x}.png"))
        patient_table["img"] = multithread_read_img(patient_table["img_paths"], (patch_shape, patch_shape))

        visualize_attention(patch_shape, model_name, patient_table, attention, save_dir, patient_id)


def main():
    config_dir = os.path.join("configs", "attention-vis-config.json")
    args = get_args(config_dir)
    mil = True if args["model"] == "attention-mil" else False

    root_data_dir = os.path.join("..", "data", args["feature_extractor"], args["embedding_type"])
    patch_dir = os.path.join("..", "..", "raw-data", "patches", f"experiment-{args['experiment_num']}")
    patient_table_dir = os.path.join("..", "..", "raw-data", "embeddings", f"experiment-{args['experiment_num']}", args["feature_extractor"])

    base_model_dir, base_save_dir = get_save_dirs(args, mode="inference")
    num_splits = len(os.listdir(root_data_dir))

    for split_num in range(1, num_splits + 1):
        print(f"Trial [{split_num}/{num_splits}]")

        trial_dir = os.path.join(root_data_dir, f"split-{split_num}")
        inference_dir = os.path.join(trial_dir, "test")

        label_dir = os.path.join("..", "data", "labels.csv")
        model_dir = os.path.join(base_model_dir, f"split-{split_num}")
        save_dir = os.path.join(base_save_dir, f"split-{split_num}", f"{args['model']}-explainability")

        os.makedirs(save_dir, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        inference_dataset = WSIDataset(inference_dir, label_dir, mil, args["pad"], False, args["embedding_type"], args["target_shape"])
        inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)

        model, save_base_name = get_model(args)
        model = model.to(device)

        weights_dir = os.path.join(model_dir, f"{save_base_name}-{args['weights']}")
        weights = torch.load(weights_dir, map_location=torch.device(device), weights_only=True)
        model.load_state_dict(weights)

        forward_pass(
            inference_loader, model, patient_table_dir, patch_dir, args["patch_shape"], save_dir, args["model"], mil, device
        )


if __name__ == "__main__":
    main()