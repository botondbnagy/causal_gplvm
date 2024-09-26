'''
Use this script to run causal analysis on the results of the experiments,
i.e. to determine the causal direction of each dataset and plot the predictions.
If experiment finished, predictions_hpc.csv should be in the results folder, and
no re-fitting is required. In this case, run only plotPredictions(), with hpc=True.
'''

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import hashlib
from sklearn.preprocessing import StandardScaler

from causal_discovery_cross_validation.gplvm_causal_discovery.methods.crossvalidationBayes import Marginal, Conditional
from utils import preprocess


def getTestLoss(comp_table, trainval_data, test_data, num_inducing, causal, pair_index, plot_fit, model):
    
    # reduce comp_table to only the relevant rows
    red_comp_table = comp_table.loc[(comp_table['dataset'] == pair_index) &
                                (comp_table['causal'] == causal) &
                                (comp_table['model'] == model)].reset_index(drop=True)
    

    if not causal:
        # flip x and y
        trainval_data = np.flip(trainval_data, axis=0)
        test_data = np.flip(test_data, axis=0)
         

    if model == 'marginal':
        trainval_data = trainval_data[1].reshape(1, -1, 1)
        test_data = test_data[1].reshape(1, -1, 1)

        loss = Marginal().getBestLoss(red_comp_table,
                                       trainval_data, 
                                       test_data, 
                                       num_inducing=num_inducing,
                                       causal=causal,
                                       pair_index=pair_index,
                                       plot_fit=plot_fit)
        
    elif model == 'conditional':
        trainval_data = trainval_data.reshape(2, -1, 1)
        test_data = test_data.reshape(2, -1, 1)

        loss = Conditional().getBestLoss(red_comp_table,
                                         trainval_data, 
                                         test_data, 
                                         num_inducing=num_inducing,
                                         causal=causal,
                                         pair_index=pair_index,
                                         plot_fit=plot_fit)


    return loss


def plotPredictions(work_dir, hpc=False, dataset_name=''):
    """
    Plot predictions of causal direction for each dataset
    """
    #set manual color cycle
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['lightseagreen', 'salmon', 'slateblue', 'gold', 'mediumorchid', '#116b67', '#9a352d', '#3f3c6f', '#7b5a00', '#6d215e'])

    figpath = work_dir + '/figs/' + dataset_name + '/'
    results_path = work_dir + '/results/' + dataset_name + '/'

    #load predictions.csv
    if hpc:
        predictions = pd.read_csv(results_path + 'predictions_hpc.csv')
    else:
        predictions = pd.read_csv(results_path + 'predictions.csv')

    #get truth for each dataset
    datasets = predictions['dataset'].values.astype(int)
    targets = predictions['target'].values
    guesses = predictions['guess'].values

    #bar plot of each dataset on x axis, with loss predictions of direction on y axis as 1 and -1, coloured by truth
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    #plot correct guesses in green, incorrect in red
    if np.sum(guesses == targets) > 0:
        ax.bar(datasets[guesses == targets], guesses[guesses == targets], color='C0', label='correct')
    if np.sum(guesses != targets) > 0:
        ax.bar(datasets[guesses != targets], guesses[guesses != targets], color='C1', label='incorrect')

    #1 is causal, -1 is anti-causal on y axis
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(['anti-causal', 'causal'])
    ax.set_xlabel('dataset')
    ax.set_ylabel('prediction')
    ax.legend()
    plt.tight_layout()
    plt.savefig(figpath + 'AN_causal_bar.jpg')
    plt.show()
    plt.close()



    #plot loss difference between causal and anti-causal for each dataset
    loss_diff = predictions['anti-causal_test_loss'].values - predictions['causal_test_loss'].values
    plt.figure(figsize=(15, 4))

    #correct guesses in green, incorrect in red
    if np.sum(guesses == targets) > 0:
        plt.stem(datasets[guesses == targets], loss_diff[guesses == targets], label='correct', markerfmt='C0o', linefmt='C0-', basefmt=' ')
    if np.sum(guesses != targets) > 0:
        plt.stem(datasets[guesses != targets], loss_diff[guesses != targets], label='incorrect', markerfmt='C1o', linefmt='C1-', basefmt=' ')
    #extend base line to cover entire x axis (sorted datasets)
    dsort = np.sort(datasets)
    plt.plot([dsort[0], dsort[-1]], [0, 0], 'k--', label='no difference')

    plt.xlabel('dataset')
    plt.ylabel('test loss difference (anti-causal - causal)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figpath + 'AN_causal_stem.jpg')
    plt.show()
    plt.close()

    print("loss prediction accuracy: {:.3f}".format(np.mean(guesses == targets)))

def getPredictions(x_all: np.ndarray,
                   y_all: np.ndarray,
                   target_all: np.ndarray,
                   test_size=0.2,
                   num_inducing=100,
                   work_dir='.'):
    
    #import comp_table (losses dataframe) from each dataset in results folder named losses_table_core...
    results_path = work_dir + '/results/'

    all_losses = pd.DataFrame()
    for file in os.listdir(results_path):
        if file.startswith('loss_table_core'):
            all_losses = pd.concat([all_losses, pd.read_csv(results_path + file)])

    comp_table = all_losses.reset_index(drop=True)

    #get number of datasets from loss table dataset column
    dataset_range = comp_table['dataset'].unique().astype(int)
    print('Found {} datasets'.format(len(dataset_range)))

    #get hashes from file
    with open(results_path + 'test_data_hashes.csv', 'r') as f:
        hashes = f.readlines()
        hashes = [h.strip().split(',') for h in hashes]
        hashes = {int(h[0]): h[1] for h in hashes}

    predictions = pd.DataFrame(columns=['dataset',
                                        'target',
                                        'causal_test_loss',
                                        'anti-causal_test_loss',
                                        'guess'])

    for pair_index in dataset_range:
        i = pair_index - 1
        x, y, target = x_all[i], y_all[i], target_all[i]
        trainval_data, test_data = preprocess(x, y, test_size=test_size, shuffle_seed=pair_index)

        print("pair {} |".format(pair_index), end=' ')
        print("target: {}".format('causal' if target == 1 else 'anti-causal'))

        #get md5 hash of test dataset
        test_C = np.ascontiguousarray(test_data) #convert to contiguous array for hashing
        test_hash = hashlib.md5(test_C).hexdigest()
        print("Test data hash: {}".format(test_hash))
        #verify that test data hash matches hash in file for this pair
        print(hashes[pair_index])
        assert test_hash == hashes[pair_index], "Test data hash does not match hash in file"
        print("Test data hash matches hash in file")
        
        #same with for loops
        total_losses = {'causal': 0, 'anti-causal': 0}
        for causal in [True, False]:
            for model in ['marginal', 'conditional']:

                test_loss = getTestLoss(comp_table,
                                    trainval_data,
                                    test_data,
                                    num_inducing=num_inducing,
                                    causal=causal,
                                    pair_index=pair_index,
                                    plot_fit=True,
                                    model=model)
                
                total_losses['causal' if causal else 'anti-causal'] += test_loss

        predictions = predictions.append({'dataset': pair_index,
                                        'target': target,
                                        'causal_test_loss': total_losses['causal'],
                                        'anti-causal_test_loss': total_losses['anti-causal'],
                                        'guess': 1 if total_losses['causal'] < total_losses['anti-causal'] else -1
                                        },
                                            ignore_index=True)

    predictions.to_csv(results_path + 'predictions.csv', index=False)
    plotPredictions(work_dir=work_dir)
