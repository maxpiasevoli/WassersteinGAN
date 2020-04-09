import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .DataPadding import DataPadding
from .SplitData import SplitData

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class BehavioralDataset(Dataset):
    """Behavioral Learning dataset."""

    def __init__(self, isCnnData, isScoring=False, auto_number=-1, output_directory='./', transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        original_wd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), 'data'))
        if isScoring:
            self.trials_df = pd.read_csv('./behavioral_cv.csv')
        else:
            full_trials_df = pd.read_csv('./behavioral.csv')
            trials_df = SplitData.holdoutSamples(full_trials_df, auto_number,
                                                 'behavioral', output_directory)
            self.trials_df = trials_df
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
        # if self.isCnnData:
        num_features = trials.shape[1]
        trials_temp = [trials for i in range(num_features)]
        trials = np.vstack(trials_temp)

        trials = trials.reshape(1, trials.shape[0], trials.shape[1])
        trials = torch.from_numpy(trials)

        if self.transform:
            trials = self.transform(trials)

        # pad data if using conv filters
        if self.isCnnData:
            trials = DataPadding.padData(trials)

        # second position should be a label however we have no use for this
        return trials, []
