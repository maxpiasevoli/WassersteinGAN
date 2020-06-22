import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from data import DataPadding
import random

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class StopAndFriskDataset(Dataset):
    """Behavioral Learning dataset."""

    def __init__(self, isCnnData, transform=None):
        """
        Args:
            isCnnData (boolean, required): if True, applies necessary padding to
                the stacked data.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Note: The "stacking" of individual samples in this Dataset class is
        different from the stacking used in BehavioralDataset.py.
        In BehavioralDataset.py, the same sample was stacked multiple times to
        form a 2D sample with equal length and width dimensions. In this class,
        samples from the entire dataset are randomly selected with replacement
        and stacked to form each 2D sample with equal length and width dimensions.
        This logic can be seen in lines 54-58. This variation of stacking was
        chosen given the lower dimensionality of the Stop-And-Frisk data. Note
        that compared to BehavioralDataset.py, this Dataset class still needs
        various features implemented like having the option to generate
        separate training and cross validation subsets for a wgan model and
        then read samples from a cross validation subset to enable calculation
        of the wins matrix and use of the generative ability model.

        """
        print("CWD")
        original_wd = os.getcwd()
        print(os.getcwd())
        os.chdir(os.path.join(os.getcwd(), 'data'))
        print(os.getcwd())
        data_file_name = '' # specify the data file name containing the correct
                            # subset of Stop-And-Friks data
        self.trials_df = pd.read_csv(data_file_name)
        self.transform = transform
        self.isCnnData = isCnnData
        os.chdir(original_wd)

    def __len__(self):
        return len(self.trials_df)

    def _fetchRow(self, idx):
        return np.array([self.trials_df.iloc[idx,0:]])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        num_samples = self.trials_df.shape[0]
        num_features = self.trials_df.shape[1]
        selected_indices = [i for i in range(num_samples)]
        random.shuffle(selected_indices)
        selected_indices = selected_indices[:num_features]
        trials = [self._fetch_row(index) for index in selected_indices]
        trials = np.vstack(trials)

        trials = trials.reshape(1, trials.shape[0], trials.shape[1])
        trials = torch.from_numpy(trials)

        if self.transform:
            trials = self.transform(trials)

        # pad data if using conv filters
        if self.isCnnData:
            trials = DataPadding.padData(trials)

        # second position should be a label however we have no use for this
        return trials, []
