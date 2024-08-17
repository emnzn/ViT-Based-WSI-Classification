import os
from typing import Dict, Union

import numpy as np
import pandas as pd

def save_embeddings(
    results: Dict[str, Union[str, np.ndarray]],
    save_dir: int
    ) -> None:

    """
    Saves the embeddings and coordinates of each patch as a parquet.

    Parameters
    ----------
    results: Dict[str, Union[str, np.ndarray]]
        A dictionary containing the coordinates and embeddings of each patch.
    """

    df = pd.DataFrame(results)
    df.to_parquet(save_dir)
