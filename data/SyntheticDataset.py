import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from data import DataPadding

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class SyntheticDataset(Dataset):
    """Behavioral Learning dataset."""

    def __init__(self, isCnnData, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("CWD")
        original_wd = os.getcwd()
        print(os.getcwd())
        os.chdir(os.path.join(os.getcwd(), 'data'))
        print(os.getcwd())
        self.trials_df = pd.read_csv('./synDist.csv')
        self.transform = transform
        self.isCnnData = isCnnData
        os.chdir(original_wd)

    def __len__(self):
        return len(self.trials_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        trials = self.trials_df.iloc[idx, 0:]
        trials = np.array([trials])

        # stack data if using conv filters
        if self.isCnnData:
            num_features = trials.shape[1]
            trials_temp = [trials for i in range(num_features)]
            trials = np.vstack(trials_temp)

        #trials = np.dstack(trials)
        trials = trials.reshape(1, trials.shape[0], trials.shape[1])
        trials = torch.from_numpy(trials)

        if self.transform:
            trials = self.transform(trials)

        # pad data if using conv filters
        if self.isCnnData:
            trials = DataPadding.padData(trials)

        # second position should be a label however we have no use for this
        return trials, []
