# CyclicLossEstimator
Using helper networks to estimate complicated loss functions 


## Usage

First, train the loss estimator network  
    python main.py --runName myLossEstimator --lossEstimator train --dataDir ../../FIDDLE/data/hdf5datasets


Then, train the inference network  
    python main.py --runName cs2ds --lossEstimator myLossEstimator --dataDir ../../FIDDLE/data/hdf5datasets

