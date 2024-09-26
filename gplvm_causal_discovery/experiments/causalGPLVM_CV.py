"""
This method fits a GPLVM to the data using cross-validation to select the best
hyperparameters. The hyperparameters are selected by minimising validation
loss. Best hyperparameters are then used to fit the model to the whole dataset
with both causal directions. The causal direction is then selected by comparing 
the test loss and marginal likelihood of the two models.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm import trange
from typing import Optional
import tensorflow as tf
import hashlib

from methods.crossvalidation_general import Marginal, Conditional
from utils import preprocess

def run(x: np.ndarray,
        y: np.ndarray,
        target: int,
        method: str,
        hyperspace: dict,
        pair_index: int,
        n_iters: int=1,
        num_inducing: int=100,
        minibatch_size: int=100,
        opt_iter: int=5000,
        test_size: float=0.2,
        work_dir: str='.',
        dataset: str='',
        random_restart_number: Optional[int]=0,
        ):
    """
    Main function for fitting GPLVM to given dataset in both causal directions

    Args:
        CORE: index of starting dataset (core for hpc)
        n_datasets: number of datasets to fit (starting from CORE)
        n_iters: number of hyperparameters to try
        test_size: proportion of data to use for testing
        work_dir: working directory
        dataset: dataset to use
        k: number of folds for cross-validation

    Returns:
        loss_df: dataframe containing test loss for each dataset
    """
    results_path = Path(work_dir) / 'results/' / dataset

    #set random seed based on random restart and dataset
    np.random.seed(1000 * random_restart_number + pair_index)
    tf.random.set_seed(1000 * random_restart_number + pair_index)

    # get current dataset and preprocess
    trainval_data, test_data = preprocess(x, y, test_size=test_size, shuffle_seed=pair_index)

    tf.print("pair {} |".format(pair_index), end=' ')
    tf.print("target: {}".format('causal' if target == 1 else 'anti-causal'))
    
    #get md5 hash of test dataset
    test_C = np.ascontiguousarray(test_data) # convert to contiguous array for hashing
    test_hash = hashlib.md5(test_C).hexdigest()
    tf.print("Test data hash: {}".format(test_hash))

    #save test data hash to file
    with open(results_path / 'test_data_hashes.csv', 'a') as f:
        #write pair index and hash to file
        f.write('{},{}\n'.format(pair_index, test_hash))
    
    pair_predictions = {'dataset': pair_index,
                        'target': target,
                        'causal_test_loss': None,
                        'anti-causal_test_loss': None,
                        'guess': None}

    # do the following in both directions (x->y and y->x)
    for causal in [True, False]:
        if causal:
            tf.print('Causal direction')
            #do nothing (leave data as is)
            
        else:
            tf.print('Anti-causal direction')
            #swap x and y
            # trainval_data = np.flip(trainval_data, axis=0)
            # test_data = np.flip(test_data, axis=0)
            trainval_data = (trainval_data[1], trainval_data[0])
            test_data = (test_data[1], test_data[0])

        # get best marginal model loss
        tf.print('Finding best marginal model hyperparameters')
        #x as shape (1, n, dim), so place in an outer array
        x_trainval = np.array([trainval_data[0]])
        x_test = np.array([test_data[0]])
        best_marg_test_loss = Marginal().randomSearch(trainval_data=x_trainval,
                                                    test_data=x_test,
                                                    hyperspace=hyperspace,
                                                    method=method,
                                                    num_inducing=num_inducing,
                                                    n_iter=n_iters,
                                                    opt_iter=opt_iter,
                                                    num_minibatch=minibatch_size,
                                                    plot_fit=True,
                                                    causal=causal,
                                                    pair_index=pair_index,
                                                    random_restart_number=random_restart_number,
                                                    dataset_name=dataset)

        # get best conditional model loss
        tf.print('Finding best conditional model hyperparameters')
        best_cond_test_loss = Conditional().randomSearch(trainval_data=trainval_data,
                                                            test_data=test_data,
                                                            hyperspace=hyperspace,
                                                            method=method,
                                                            num_inducing=num_inducing,
                                                            n_iter=n_iters,
                                                            opt_iter=opt_iter,
                                                            num_minibatch=minibatch_size,
                                                            plot_fit=True,
                                                            causal=causal,
                                                            pair_index=pair_index,
                                                            random_restart_number=random_restart_number,
                                                            dataset_name=dataset)
        

        total_test_loss = best_marg_test_loss + best_cond_test_loss

        #save predictions to dictionary
        pair_predictions['causal_test_loss' if causal else 'anti-causal_test_loss'] = total_test_loss
        
    #write results to file predictions_hpc.csv (create file if it doesn't exist)
    pair_predictions['guess'] = 1 if pair_predictions['causal_test_loss'] < pair_predictions['anti-causal_test_loss'] else -1
    new_row = pd.DataFrame(pair_predictions, index=[0])
    fname = results_path / 'predictions.csv'
    new_row.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

    tf.print('Finished pair {}'.format(pair_index))




            
