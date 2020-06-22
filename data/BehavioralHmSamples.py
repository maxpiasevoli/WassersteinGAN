import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .DataPadding import DataPadding

# Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class BehavioralHmSamples(Dataset):
    """Behavioral hierarchical models samples dataset."""

    def __init__(self, modelNum, isCnnData, isScoring, transform=None):
        """
        Args:
            modelNum (int, required): specifies which hierarchical model number to draw samples
                from.
            isCnnData (boolean, required): if True, applies necessary padding to the stacked data.
            isScoring (boolean, required): if True, returns samples from the sample set containing
                1000 different samples. Otherwise returns samples from the sample
                set containing 30 different samples. Have this parameter be true
                when constructing the wins matrix to be used in the generative
                ability model.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        #print('please, ', os.getcwd())
        original_wd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), 'data'))
        if isScoring:
            if modelNum == 1:
                dataset_path = './bl_m1_1000.csv'
            elif modelNum == 2:
                dataset_path = './bl_m2_1000.csv'
            elif modelNum == 3:
                dataset_path = './bl_m3_1000.csv'
            elif modelNum == 4:
                dataset_path = './bl_m4_1000.csv'
            else:
                dataset_path = './bl_m5_1000.csv'
        else:
            if modelNum == 1:
                dataset_path = './bl_m1.csv'
            elif modelNum == 2:
                dataset_path = './bl_m2.csv'
            elif modelNum == 3:
                dataset_path = './bl_m3.csv'
            elif modelNum == 4:
                dataset_path = './bl_m4.csv'
            else:
                dataset_path = './bl_m5.csv'
        #print(os.getcwd())
        self.trials_df = pd.read_csv(dataset_path)
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
            trials = DataPadding.padData(trials, trials.shape[1], trials.shape[2])

        # second position should be a label however we have no use for this
        return trials, []
