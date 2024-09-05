import os
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class WSIDataset(Dataset):

    """
    Creates the dataset class for the dataloader.

    Parameters
    ----------
    data_dir: str
        The directory to the embeddings.
    
    label_dir: str
        The directory to the labels.

    mil: bool
        Whether compiling for a Multiple-Instance Based Model.

    pad: bool
        Wether to pad inputs to a pre-determined size.

    embedding_type: str
        One of [stitched, isolated].

        `Isolated` refers to embeddings that have not been stitched back together.
        This means that each instance is an embedding of a tissue region.

        Alternatively, `stitched` refers to embeddings that maintain the spatial relationships
        found in the original WSI. The embedding of each instance is stitched together to 
        create a smaller representation of the entire WSI, whereby the embedding is placed at
        the channel dimension.

    target_shape: List[int]
        The target shape to pad images into.
        Must be in the form (width, height).

    Returns
    -------
    embedding: torch.Tensor
        The embedding of the WSI given a foundation model.
    
    label: str
        The grade of the patient at the given datapoint.

    patient_id: str
        The patient id.
    """

    def __init__(
        self, 
        data_dir: str, 
        label_dir: str,
        mil: bool,
        pad: bool,
        embedding_type: str,
        target_shape: List[int]
        ):

        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        self.labels = self.generate_labels(label_dir)
        self.mil = mil
        self.pad = pad
        self.embedding_type = embedding_type
        self.target_shape = target_shape

        assert all([Path(i).stem in self.labels for i in self.filenames]), "All patient ids must have a label"

    
    def generate_labels(self, label_dir: str) -> Dict[str, str]:

        """
        Creates a dictionary containing the patient ids as keys
        and the associated Meningioma grade as the values.
        """

        labels = pd.read_csv(label_dir)
        ids = labels["id"].tolist()
        grades = labels["grade"].map(lambda x: 0 if x == "1" else 1).tolist()

        labels = {patient_id: grade for patient_id, grade in zip(ids, grades)}

        return labels

    def pad_embedding(
        self, 
        embedding: torch.Tensor, 
        target_shape: List[int]
        ) -> np.ndarray:

        """
        Pads the embedding to a target shape.
        The tensor must be of shape [C, H, W]
        """

        current_shape = embedding.shape[1:]

        delta_h = target_shape[0] - current_shape[0]
        delta_w = target_shape[1] - current_shape[1]

        pad_top = delta_h // 2
        pad_bottom = delta_h - pad_top
        
        pad_left = delta_w // 2
        pad_right = delta_w - pad_left

        m = torch.nn.ZeroPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))

        padded_embedding = m(embedding)

        return padded_embedding
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        patient_id = Path(filename).stem
        label = self.labels[patient_id]

        embedding_path = os.path.join(self.data_dir, filename)

        if self.embedding_type == "isolated":
            embedding = torch.tensor(np.load(embedding_path))

        else:
            embedding = torch.tensor(np.load(embedding_path)).permute(2, 0, 1) # [channels, height, width]

            if self.pad:
                embedding = self.pad_embedding(embedding, self.target_shape)

            if self.mil:
                channels, height, width = embedding.shape
                embedding = embedding.permute(1, 2, 0).reshape(height * width, channels)

        return embedding, label, patient_id

