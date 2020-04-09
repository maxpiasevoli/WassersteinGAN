import numpy as np
import pandas as pd
import random

class SplitData:

    # splits data into training and cross validation sets.
    # outputs training and cross validation sets
    @staticmethod
    def holdoutSamples(df, automation_number, dataset_name, output_directory):

        # randomize indices
        indices = [i for i in range(len(df))]
        random.shuffle(indices)
        cv_indices = indices[:3]

        # output training and cross validation sets
        cv_samples = df.iloc[cv_indices]
        df = df.drop(cv_indices)
        df.to_csv('{0}/{1}_{2}_training.csv'.format(output_directory, dataset_name, automation_number))
        cv_samples.to_csv('{0}/{1}_{2}_cv.csv'.format(output_directory, dataset_name, automation_number))

        return df
