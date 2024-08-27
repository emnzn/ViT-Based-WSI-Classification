import os
from typing import Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils import (
    WSIDataset, get_args, save_results,
    resnet, attention_mil, swin_transformer,
    ResNet, SwinTransformer, AttentionBasedMIL,
)

@torch.no_grad()
def inference(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    mil: bool,
    device: str,
    save_dir: str,
    save_filename: str
    ) -> Tuple[float, float, float]:

    """
    Runs inference on a given model.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    criterion: nn.Module
        The loss function.

    model: Union[ResNet, SwinTransformer, AttentionBasedMIL]
        The model to be trained.

    mil: bool
        Whether training under a Multiple-Instance Learning scheme.
        This is used because the Attenion-Based MIL model returns the 
        attention weights placed on each instance.

    device: str
        One of [cuda, cpu].

    save_dir: str
        The directory to save the results. 
        The results will be saved as a csv file where the prediction
        for each patient id can be accessed.

    save_filename: str
        The filename to use when saving the results.


    Returns
    -------
    average_loss: float
        The average loss across all samples.

    average_f1: float
        The average weighted f1 across all samples.

    average_balanced_accuracy: float
        The average balanced accuracy across all samples.
    """
    
    metrics = {
        "patient_id": [],
        "loss": [],
        "prediction": [],
        "target": []
    }

    model.eval()
    for wsi_embedding, target, patient_id in tqdm(dataloader, desc="Validation in progess"):
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if mil: 
            logits, _ = model(wsi_embedding)

        else: 
            logits = model(wsi_embedding)

        loss = criterion(logits, target)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["patient_id"].extend(patient_id)
        metrics["loss"].append(loss.detach().cpu().item())
        metrics["prediction"].extend(pred.cpu().numpy())
        metrics["target"].extend(target.cpu().numpy())

    average_loss = sum(metrics["loss"]) / len(dataloader)
    average_f1 = f1_score(metrics["target"], metrics["prediction"], average="weighted")
    average_balanced_accuracy = balanced_accuracy_score(metrics["target"], metrics["prediction"])

    save_results(metrics, save_dir, save_filename)
    
    return average_loss, average_f1, average_balanced_accuracy


def main():
    config_dir = os.path.join("configs", "inference-config.yaml")
    args = get_args(config_dir)
    mil = True if args["model"] == "attention-mil" else False

    root_data_dir = os.path.join("..", "data", args["feature_extractor"], f"trial-{args['trial_num']}")
    inference_dir = os.path.join(root_data_dir, "test")

    label_dir = os.path.join("..", "data", "labels.csv")
    model_dir = os.path.join("..", "assets", "model-weights", f"trial-{args['trial_num']}")
    save_dir = os.path.join("..", "assets", "inference-results", f"trial-{args['trial_num']}")

    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference_dataset = WSIDataset(data_dir=inference_dir, label_dir=label_dir, mil=mil)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    if mil:
        model = attention_mil(num_classes=args["num_classes"]).to(device)
        model_base_name = args["model"]

    if args["model"] == "resnet":
        model = resnet(
            variant=args["variant"], num_classes=args["num_classes"]
            ).to(device)
        
        model_base_name = f"{args['variant']}"

    if args["model"] == "swin":
        model = swin_transformer(
            version=args["version"], variant=args["variant"], dropout=args["dropout_probability"], num_classes=args["num_classes"]
            ).to(device)
        
        model_base_name = f"{args['model']}-{args['version']}-{args['variant']}"

    weights_dir = os.path.join(model_dir, f"{model_base_name}-{args['weights']}")
    weights = torch.load(weights_dir, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(weights)

    criterion = nn.CrossEntropyLoss()

    average_loss, average_f1, average_balanced_accuracy = inference(
        dataloader=inference_loader, criterion=criterion, model=model, mil=mil,
        device=device, save_dir=save_dir, save_filename=model_base_name
    )

    print("Inference Statistics:")
    print(f"Loss: {average_loss:.4f} | F1 Score: {average_f1:.4f} | Balanced Accuracy: {average_balanced_accuracy:.4f}\n")


if __name__ == "__main__":
    main()