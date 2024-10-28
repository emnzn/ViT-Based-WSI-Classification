import os
from typing import Dict, List

import numpy as np
import pandas as pd

def save_results(
    data_table: Dict[str, List[np.ndarray]], 
    save_dir: str,
    save_filename: str,
    mode: str = "metrics"
    ):

    valid_modes = ["metrics", "attention"]

    assert mode in valid_modes, f"mode must be one of f{valid_modes}"

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(data_table)
    if mode == "metrics":
        df.to_csv(os.path.join(save_dir, f"{save_filename}-results.csv"), index=False)

    elif mode == "attention":
        df.to_parquet(os.path.join(save_dir, f"{save_filename}-instance-weights.parquet"), index=False)