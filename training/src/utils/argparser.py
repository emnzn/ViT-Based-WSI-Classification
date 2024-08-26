import os
import yaml
from typing import Dict, Union

def get_args(arg_dir: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    arg_dir: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_dir, "r") as f:
        args = yaml.safe_load(f)

    return args

def save_args(log_dir: str, args: Dict[str, Union[float, str]]) -> None:
    """
    Saves arguments inside a log directory.

    Parameters
    ----------
    log_dir: str
        The destination directory to save the arguments to.

    args: Dict[str, Union[float, str]]
        The arguments to be saved. The resulting yaml file will have a filename `run_config.yaml`.
    """

    path = os.path.join(log_dir, "run_config.yaml")

    training = {
        "seed" : args["seed"],
        "model": args["model"],
        "epochs": args["epochs"],
        "eta_min": args["eta_min"],
        "trial_num": args["trial_num"],
        "num_classes": args["num_classes"],
        "learning_rate": args["learning_rate"],
        "feature_extractor": args["feature_extractor"],
        "grad_accumulation": args["grad_accumulation"]
        }
    
    regularization = {
        "weight_decay": args["weight_decay"],
        "label_smoothing": args["label_smoothing"],
        "dropout_probability": args["dropout_probability"]
    }

    model = {
        "variant": args["variant"],
        "version": args["version"]
    }

    organized_args = {
        "training": training,
        "regularization": regularization,
        "model": model
    }
    
    with open(path, "w") as f:
        yaml.dump(organized_args, f)