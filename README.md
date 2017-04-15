# CyclicLossEstimator
This project is a subproject for [FIDDLE](https://github.com/ueser/FIDDLE).  

The aim is to use helper networks to estimate complicated loss functions.


## Usage

First, train the loss estimator network  
    python main.py --runName myLossEstimator --lossEstimator train --dataDir ../../FIDDLE/data/hdf5datasets

Then, train the inference network  
    python main.py --runName cs2ds --lossEstimator myLossEstimator --dataDir ../../FIDDLE/data/hdf5datasets

## An example for jointly infering regulatory code and its location
Here, I first trained the loss estimator by providing DNA sequence to predict TSS-seq data using KL divergence as the loss function.  
Then, I trained the DNA sequence inference network by providing TSS-seq data. But, here, instead of using KL divergence, I used loss estimator network to backpropagate its loss to the inference network by freezing the weights of loss estimator. Resulting is the prediction of DNA sequence which minimizes the TSS-seq prediction loss. 

GIFs below show how the prediction for 5 random datapoints evolves over iterations (every iteration is 1000 sample).
The top one is the TSS-seq prediction as a result of loss-estimator and the bottom one is the output of the inference network. 

![alt text](https://cloud.githubusercontent.com/assets/1741502/24807507/76fa3ad8-1b86-11e7-8400-999a7c593188.gif)

![alt text](https://cloud.githubusercontent.com/assets/1741502/24806673/9a72a6ba-1b83-11e7-8142-d4ab8ea09dcb.gif)

The following GIF shows the prediction of important regulatory DNA sequence from CTCF ChIP-seq data along with the reconstructed ChIP-seq signal. Note that the reconstruction is paid off by the regularizing DNA sequence to be interpretable. 

![alt_text](https://cloud.githubusercontent.com/assets/1741502/25063871/babd5188-21bc-11e7-86de-b0c218f90c71.gif)
