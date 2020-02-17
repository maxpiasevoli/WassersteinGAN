import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class BehavioralDataset(Dataset):
    """Behavioral Learning dataset."""

    def __init__(self, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trials_df = pd.read_csv('./behavioral.csv')
        self.transform = transform

    def __len__(self):
        return len(self.trials_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        trials = self.trials_df.iloc[idx, 1:]
        trials = np.array([trials]).reshape((5,5))

        if self.transform:
            sample = self.transform(sample)

        # second position should be a label however we have no use for this
        return sample, None
