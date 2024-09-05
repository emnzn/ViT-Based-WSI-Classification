import os
import json
from typing import Dict, Union

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

def save_args(log_dir: str, args: Dict[str, Union[float, str]]) -> None:
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
            "dropout_probability": args["dropout_probability"]
        },
        "model": {
            "model": args["model"],
            "variant": args["variant"],
            "version": args["version"],
            "normalization_method": args["normalization_method"]
        }
    }

    with open(path, "w") as f:
        json.dump(organized_args, f, indent=4)