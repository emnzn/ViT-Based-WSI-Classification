import os
from typing import Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils import (
    WSIDataset,
    get_args,
    save_args,
    set_seed,
    get_checkpoint,
    resnet,
    attention_mil,
    swin_transformer,
    ResNet,
    SwinTransformer,
    AttentionBasedMIL
)

def train(
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    mil: bool,
    device: str,
    grad_accumulation: int
    ) -> Tuple[float, float, float]:

    metrics = {
        "running_loss": 0,
        "running_f1": 0,
        "running_balanced_accuracy": 0,
    }

    model.train()
    for i, (wsi_embedding, target, _) in enumerate(tqdm(dataloader, desc="Training in progress")):
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if mil:
            logits, _ = model(wsi_embedding)

        else:
            logits = model(wsi_embedding)

        loss = criterion(logits, target) / grad_accumulation
        loss.backward()

        if (i + 1) % grad_accumulation == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        balanced_accuracy = balanced_accuracy_score(target.cpu(), pred.cpu())
        f1 = f1_score(target.cpu(), pred.cpu(), average="weighted")

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["running_f1"] += f1
        metrics["running_balanced_accuracy"] += balanced_accuracy

    epoch_loss = metrics["running_loss"] / (len(dataloader) / grad_accumulation)
    epoch_f1 = metrics["running_f1"] / len(dataloader)
    epoch_balanced_accuracy = metrics["running_balanced_accuracy"] / len(dataloader)

    return epoch_loss, epoch_f1, epoch_balanced_accuracy


@torch.no_grad()
def validate(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL],
    mil: bool,
    device: str
    ) -> Tuple[float, float, float]:
    
    metrics = {
        "running_loss": 0,
        "running_f1": 0,
        "running_balanced_accuracy": 0,
    }

    model.eval()
    for wsi_embedding, target in tqdm(dataloader, desc="Validation in progess"):
        wsi_embedding = wsi_embedding.to(device)
        target = target.to(device)

        if mil: 
            logits, _ = model(wsi_embedding)

        else: 
            logits = model(wsi_embedding)

        loss = criterion(logits, target)
        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        balanced_accuracy = balanced_accuracy_score(target.cpu(), pred.cpu())
        f1 = f1_score(target.cpu(), pred.cpu(), average="weighted")

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["running_f1"] += f1
        metrics["running_balanced_accuracy"] += balanced_accuracy

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_f1 = metrics["running_f1"] / len(dataloader)
    epoch_balanced_accuracy = metrics["running_balanced_accuracy"] / len(dataloader)
    
    return epoch_loss, epoch_f1, epoch_balanced_accuracy

def main():
    pass