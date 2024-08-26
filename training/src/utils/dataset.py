import os
from typing import Dict
from pathlib import Path

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
    """

    def __init__(
        self, 
        data_dir: str, 
        label_dir: str,
        mil: bool
        ):

        self.data_dir = data_dir
        self.filenames = os.listdir(data_dir)
        self.labels = self.generate_labels(label_dir)
        self.mil = mil

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
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        patient_id = Path(filename).stem
        label = self.labels[patient_id]

        embedding_path = os.path.join(self.data_dir, filename)
        embedding = torch.tensor(np.load(embedding_path)) # [height, width, channels]

        if self.mil:
            height, width, channels = embedding.shape
            embedding = embedding.reshape(height * width, channels)

        else:
            embedding = embedding.permute(2, 0, 1) # [channels, height, width]

        return embedding, label, patient_id
