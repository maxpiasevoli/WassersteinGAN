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

This notebook contains the code that performs all preprocessing to generate the data that will be used to recreate the Stop-And-Frisk Hierarchical Models.

### Network_Check.ipynb

This notebook contains code that for the behavioral learning data visualizes fake samples from the generator, visualizes critic score distributions for all real and fake sample sets, uses Platt Scaling on the critic scores of real and fake sample sets, and visualizes filters for the critic. Though I have written this code both for the convolutional architecture WGAN (referred to in the notebook as dcgan) and for the ordinary feed forward network WGAN (referred to in the notebook as MLP), only the results for the dcgan variation are reliable at this point since the MLP variant still needs to have its structure and hyperparameters tuned. The dcgan variant is not perfectly tuned however it does a reasonably well job of learning the behavioral learning dataset.     

### Scoring.ipynb

This notebook calculates the wins matrix that is later used for the generative ability model and also visualizes the average win percentage for each hierarchical model across multiple WGANs.

## R files

### H_Models_BL_SF.R

Contains all hierarchical models for the behavioral learning data and Stop-and-Frisk data as well. Note that the first two behavioral learning models were proposed in the Andrew Gelman and Jennifer Hill textbook, [Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/) while the three Stop-and-Frisk models were taken from the Gelman, Fagan and Kiss paper, [*An Analysis of the New York City Police Department's "Stop-and-Frisk" Policy in the Context of Claims of Racial Bias*](http://www.stat.columbia.edu/~gelman/research/published/frisk9.pdf).

### Score_Analysis.R

Reads in the produced wins matrix produced by Scoring.ipynb and runs the generative ability model.

## Data Sources

### Stop-and-Frisk

Regarding the data for the Stop-and-Frisk models, we calculate the number of stops and arrests using records of stops from the [NYPD's Stop, Question, and Frisk Database](https://www1.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page). In this case, we calculate the number of stops by precinct and race in the 15 month period from January 2015 through March 2016 as well as the number of arrests in 2014. As in the Gelman paper, we consider the three races of White, Hispanic, and Black in considering stops and arrests. The Stop, Question and Frisk Database classifies stopped individuals as either White-Hispanic or Black-Hispanic, so we consider White-Hispanic individuals as "Hispanic" and Black-Hispanic individuals as "Black" for this data. The paper separately considers stops and arrests for violent, property, drug, and weapon crimes so we separately process stops and arrests for each type of crime.

The analysis in the Gelman paper also requires the White, Hispanic, and Black populations in each precinct for some models. To produce this data, we first downloaded the NYPD police precinct boundaries from the [NYC Open Data Portal](https://data.cityofnewyork.us/Public-Safety/Police-Precincts/78dh-3ptz). We then used the census block group boundaries from [Simply Analytics](https://app.simplyanalytics.com/index.html) and used the spatial join function to create relationship between NYC Police precincts boundaries and census block groups. Finally, the Hispanic 2017 block group level data of NYC from the [American Community Survey 2013-2017 (5-years estimate)](https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes/2017/5-year.html) which we accessed from [Social Explorer](https://www.socialexplorer.com/explore-maps) was joined to the spatial join data to calculate the populations by precinct and ethnic group. This data is stored in NYC_BlockGroups_Police_Precincts_Hispanic_Pop.csv.

The processed data is stored in multiple output files. In stop_and_frisk_data_by_precinct.csv, we have for each precinct all calculated stops and arrests for all four types of crimes as well as the populations by demographic. Then, for each type of crime, we have a .csv file for the number of arrests in the previous year as well as the number of stops in the 15 month period by precinct and race. These files respectively have the format of 2014_arrests_\*crime type\*.csv and 20152016_stops_\*crime type\*.csv.

### Behavioral Learning Experiment

The behavioral learning experiment data stored in ./data/dogs.dat is provided through the Andrew Gelman and Jennifer Hill textbook, [Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/). We then converted this data into a more amenable form which is stored in behavioral.csv.  
