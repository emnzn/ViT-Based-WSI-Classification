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
    """

    def __init__(
        self, 
        data_dir: str, 
        label_dir: str,
        ):

        self.data_dir = data_dir
        self.patient_ids = os.listdir(data_dir)
        self.labels = self.generate_labels(label_dir)

        assert all([Path(i).stem in self.labels for i in self.patient_ids]), "All patient ids must have a label"

    
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
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[Path(patient_id).stem]

        embedding_path = os.path.join(self.data_dir, patient_id)
        embedding = torch.tensor(np.load(embedding_path)).permute(2, 0, 1)

        return embedding, label
