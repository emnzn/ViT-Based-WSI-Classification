import os
import json
from typing import Dict, Union

import torch
import torch.nn as nn

from .loss import FocalLoss


def get_args(arg_dir: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a JSON file.

    Parameters
    ----------
    arg_dir: str
        The path to the JSON file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_dir, "r") as f:
        args = json.load(f)

    return args


def get_criterion(args: Dict[str, Union[float, str]]) -> nn.Module:
    
    """
    Returns the criterion for training an validation based on a coinfig file.
    """

    valid_losses = ["cross-entropy", "focal-loss"]
    assert args["loss"] in valid_losses, f"loss must be one of {valid_losses}"

    if args["loss"] == "cross-entropy":
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
        val_criterion = nn.CrossEntropyLoss()

    if args["loss"] == "focal-loss":
        alpha = torch.tensor(args["alpha"]) if args["alpha"] != None else args["alpha"]
        train_criterion = FocalLoss(alpha=alpha, gamma=args["gamma"])
        val_criterion = FocalLoss(alpha=alpha, gamma=args["gamma"])

    return train_criterion, val_criterion


def save_args(log_dir: str, args: Dict[str, Union[float, str]]):
    """
    Saves arguments inside a log directory.

    Parameters
    ----------
    log_dir: str
        The destination directory to save the arguments to.

    args: Dict[str, Union[float, str]]
        The arguments to be saved. The resulting JSON file will have a filename `run_config.json`.
    """

    path = os.path.join(log_dir, "run_config.json")

    organized_args = {
        "dataset": {
            "pad": args["pad"],
            "augment": args["augment"],
            "split_num": args["split_num"],
            "target_shape": args["target_shape"],
            "embedding_type": args["embedding_type"]
        },
        "training": {
            "seed": args["seed"],
            "epochs": args["epochs"],
            "eta_min": args["eta_min"],
            "batch_size": args["batch_size"],
            "num_classes": args["num_classes"],
            "learning_rate": args["learning_rate"],
            "feature_extractor": args["feature_extractor"],
            "grad_accumulation": args["grad_accumulation"]
        },
        "regularization": {
            "weight_decay": args["weight_decay"],
            "label_smoothing": args["label_smoothing"],
            "swin_dropout_probability": args["swin_dropout_probability"]
        },
        "model": {
            "model": args["model"],
            "resnet_normalization_method": args["resnet_normalization_method"]
        }
    }

    with open(path, "w") as f:
        json.dump(organized_args, f, indent=4)


def get_save_dirs(
    args: Dict[str, Union[float, str]],
    mode: str
    ):
    
    if mode == "train":
        if args["embedding_type"] == "isolated":
            if args["grad_accumulation"] > 1:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "grad-accumulation", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "grad-accumulation", f"split-{args['split_num']}", args["model"])

            else:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-grad-accumulation", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "no-grad-accumulation", f"split-{args['split_num']}", args["model"])


        elif args["embedding_type"] == "stitched":
            if args["augment"]:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "aug", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "aug", f"split-{args['split_num']}", args["model"])

            else:
                model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-aug", f"split-{args['split_num']}")
                log_dir = os.path.join("runs", args["embedding_type"], args["loss"], "no-aug", f"split-{args['split_num']}", args["model"])


        return model_dir, log_dir
    
    if mode == "inference":
        if args["embedding_type"] == "isolated":
            if args["grad_accumulated"]:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "grad-accumulation")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "grad-accumulation")

            else:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-grad-accumulation")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "no-grad-accumulation")

        elif args["embedding_type"] == "stitched":
            if args["augmented"]:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "aug")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "aug")

            else:
                base_model_dir = os.path.join("..", "assets", "model-weights", args["embedding_type"], args["loss"], "no-aug")
                base_save_dir = os.path.join("..", "assets", "inference-results", args["embedding_type"], args["loss"], "no-aug")

        return base_model_dir, base_save_dir