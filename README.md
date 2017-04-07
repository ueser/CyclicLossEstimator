# CyclicLossEstimator
Using helper networks to estimate complicated loss functions 


## Usage

First, train the loss estimator network  
    python main.py --runName myLossEstimator --lossEstimator train --dataDir ../../FIDDLE/data/hdf5datasets


Then, train the inference network  
    python main.py --runName cs2ds --lossEstimator myLossEstimator --dataDir ../../FIDDLE/data/hdf5datasets

## Example for jointly infering regulatory code and the location

![alt text](https://cloud.githubusercontent.com/assets/1741502/24806673/9a72a6ba-1b83-11e7-8142-d4ab8ea09dcb.gif)
