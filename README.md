Evaluating Hierarchical Models Using the WGAN
===============

The code to run the WGAN itself is in main.py. Note that for all python files in this repository, the command line arguments for the given program can be further explained by running "python *file name*.py -h". Two examples of valid commands are commented in lines 31-35 of main.py.    

In terms of output files, main.py outputs two .pth files of the format 'netG_*experiment name*.pth' for the generator and 'netD_*experiment name*.pth' for the critic. These .pth files can be used to load both trained networks at a later point. Also, main.py outputs the loss curve of the wgan in the 'w_loss_*experiment name*.png' file.  

Note that the implementation of main.py uses the PyTorch DataLoader class to load batches of samples concurrently. DataLoader requires a class that inherits the Dataset class and implements the __len__ and __getitem__ functions as specified in the documentation for each dataset that you want to train a wgan on. Examples for the behavioral learning and Stop-And-Frisk datasets are respectively shown in BehavioralDataset.py and StopAndFriskDataset.py. Note that StopAndFriskDataset.py still has features that need to be implemented that are implemented in BehavioralDataset.py. I also used classes that inherit the Dataset class to read generated samples from the hierarchical models, stack them if needed, and add necessary padding as shown in BehavioralHmSamples.py.

## GPU Helper Files

The several helper files below were written for convenience of automating runs of the WGAN and moving output files.

### trainMultWgans.py

This program automates the submission of a slurm script to execute main.py for a set number of times. Currently, the file only supports executing main.py only with the variation of the wgan that uses the convolutional architecture however this can be easily changed. The file notes which directory to save output files to given the selected gpu cluster since the home directory has limited memory.

Note that the slurm jobs usually finished with a FAILED message and exit code 137, and I think this is likely due to improperly set parameters for the slurm job. Despite this FAILED message, the WGAN model has successfully finished training if you find the output files in the designated directory.    

Generally, I advise scheduling slurm jobs manually to execute main.py until the best wgan model parameters (training iterations, learning rate, etc.) are found for a given dataset. Then, once the best model parameters have been found for a given dataset, use trainMultWgans.py to train multiple instances of that model. These models will each differ in their weight initializations.   

### cancelSlurmJobs.py

This program can be used to cancel multiple slurm jobs.

### copyGanFiles.py

This program copies all output files for all models trained by a call of trainMultWgans.py to ./WassersteinGAN in the user's home directory.  

## Jupyter Notebooks

### ASCII Art for Behavioral Learning Data.ipynb

This notebook contains the code that generates "ASCII Art" meaning visual representations of the various behavioral learning models.

### Stop and Frisk Data Preprocessing.ipynb

This notebook contains the code that performs all preprocessing to generate the data that will be used to recrete the Stop-And-Frisk Hierarchical Models.

- Stop and Frisk Data Preprocessing --> update at end

### Network_Check.ipynb

This notebook contains code that for the behavioral learning data visualizes fake samples from the generator, visualizes critic score distributions for all real and fake sample sets, uses Platt Scaling on the critic scores of real and fake sample sets, and visualizes filters for the critic. Though I have written this code both for the convolutional architecture WGAN (referred to in the notebook as dcgan) and for the ordinary feed forward network WGAN (referred to in the notebook as MLP), only the results for the dcgan variation are reliable at this point since the MLP variant still needs to have its structure and hyperparameters tuned. The dcgan variant is not perfectly tuned however it does a reasonably well job of learning the behavioral learning dataset.     

### Scoring.ipynb

This notebook calculates the wins matrix that is later used for the generative ability model and also visualizes the average win percentage for each hierarchical model across multiple WGANs.

## R files

### H_Models_BL_SF.R

Contains all hierarchical models for the behavioral learning data and Stop-and-Frisk data as well.

### Score_Analysis.R

Reads in the produced wins matrix produced by Scoring.ipynb and runs the generative ability model.

- Finish H_Models_BL_SF after stop and frisk data is updated
- Update data files after Stop-And-Frisk is sorted out
