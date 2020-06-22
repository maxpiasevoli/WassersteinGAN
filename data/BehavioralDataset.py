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

    def __init__(self, isCnnData, isScoring=False, auto_number=-1, niter=-1, output_directory='./', transform=None, useAllData=False):
        """
        Args:
            isCnnData (boolean, required): if True, applies necessary padding to the stacked data.
            isScoring (boolean, optional): if True, samples from the cross validation subset of a
                specified, trained wgan model will be provided.
            autoNumber (int/string, optional): number of the wgan model use in referring to the
                cross validation subset.
            niter (int/string, optional): thousands of iterations used to train the wgan model also used
                in referring to the cross validation subset.
            output_directory (string, optional): where to output the training and cross validation
                subsets when training a new wgan model.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            useAllData (boolean, optional): draws samples from the entire behavioral learning dataset.
                Separate training and cross validation sets are not produced.
                Note that isScoring must be false in order for this to occur.

        Note: if isScoring and useAllData are both false, then the entire behavioral
        learning dataset will be split into training and cross validation subsets.
        Samples are randomly assigned to both sets. You'll want to generate
        the training and cross validation sets when training multiple versions of
        the same wgan model to calculate the wins matrix and later use the
        generative ability model to evaluate the hierarchical models.
        """
        original_wd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), 'data'))
        if isScoring:
            self.trials_df = pd.read_csv('./behavioral_{0}k_{1}_cv.csv'.format(niter, auto_number))
        elif useAllData:
            self.trials_df = pd.read_csv('./behavioral.csv')
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
            trials = DataPadding.padData(trials, trials.shape[1], trials.shape[2])

        # second position should be a label however we have no use for this
        return trials, []
