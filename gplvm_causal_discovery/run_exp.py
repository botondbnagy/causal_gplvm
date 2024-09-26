import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from scipy import stats

from data.get_data import get_an_pairs_dataset
from data.get_data import get_tueb_dataset
from data.get_data import get_cha_dataset
from data.get_data import get_inf_data
from experiments import causalGPLVM_CV
from experiments import inf_data_optGPLVM

def run_experiment(args):
    #test if CUDA and GPU available
    tf.print("CUDA available: ", tf.test.is_built_with_cuda())
    tf.print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' #set GPU to use

    # tf.config.run_functions_eagerly(True)
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #set manual color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['lightseagreen', 'salmon', 'slateblue', 'gold', 'mediumorchid', '#116b67', '#9a352d', '#3f3c6f', '#7b5a00', '#6d215e'])
    
    #if on HPC, set core number from environment variable, otherwise use command line argument
    CORE = int(os.environ['PBS_ARRAY_INDEX']) if 'PBS_ARRAY_INDEX' in os.environ else args.core

    # arguments set from command line
    WORK_DIR = args.work_dir #working directory
    DATA = args.data #dataset to use
    K = args.n_folds #number of folds for cross-validation
    N_DATASETS = args.n_datasets #number of datasets to fit per core
    N_ITERS = args.n_iters #number of hyperparameter settings to try (or opt_iter)
    SET_SIZE = args.set_size #number of data points to use (inf_data only)
    TEST_SIZE = args.test_size #proportion of data to use for testing
    RRN = args.random_restart_num #random restart seed
    METHOD = args.method #method to use for fitting
    OPTIMISER = args.optimiser #optimiser to use for fitting
    NUM_INDUCING = args.num_inducing #number of inducing points
    MINIBATCH_SIZE = args.minibatch_size #minibatch size for optimiser

    #set hyperparameter search space
    hyperspace = {
        'sq_exp_var': stats.gamma(a=1, loc=0, scale=1),
        'sq_exp_len_obs': stats.gamma(a=1, loc=0, scale=1),
        'sq_exp_len_lat': stats.uniform(loc=0, scale=5),
        'lin_var': stats.gamma(a=1, loc=0, scale=1),
        'mat_var': stats.gamma(a=1, loc=0, scale=1),
        'mat_len_obs': stats.gamma(a=1, loc=0, scale=1),
        'mat_len_lat': stats.uniform(loc=0, scale=5),
        'rquad_var': stats.gamma(a=1, loc=0, scale=1),
        'rquad_len_obs': stats.gamma(a=1, loc=0, scale=1),
        'rquad_len_lat': stats.uniform(loc=0, scale=5),
        'likelihood_variance': stats.gamma(a=1, loc=1e-4, scale=0.1),
        'k': stats.norm(loc=K, scale=0)
        }
    
    #for each dataset assigned to this core
    for i in range(CORE * N_DATASETS, (CORE + 1) * N_DATASETS):
        pair_index = i + 1
    
        if DATA == 'an_pairs':
            tf.print("Loading AN pair {}...".format(pair_index))
            x, y, weight, target = get_an_pairs_dataset(data_path=f"{WORK_DIR}/data/an_pairs/files",
                                                        pair_index=pair_index)
            
            METHOD = 'bayes'

        elif DATA == 'tueb':
            tf.print("Loading CE-Tueb (1D) pair {}...".format(pair_index))

            x, y, weight, target, pair_index = get_tueb_dataset(data_path=f"{WORK_DIR}/data/tueb/files",
                                                    pair_index=pair_index)
            tf.print("Actual pair index: {}".format(pair_index))

            METHOD = 'general'

        elif DATA == 'cha':
            tf.print("Loading CE-Cha pair {}...".format(pair_index))
            x, y, weight, target = get_cha_dataset(data_path=f"{WORK_DIR}/data/cha/files",
                                                pair_index=pair_index)
            METHOD = 'bayes'

        #run experiment
        if DATA in ['an_pairs', 'cha', 'tueb']:
            causalGPLVM_CV.run(
                            x=x,
                            y=y,
                            target=target,
                            pair_index=pair_index,
                            hyperspace=hyperspace,
                            n_iters=N_ITERS,
                            num_inducing=NUM_INDUCING,
                            minibatch_size=MINIBATCH_SIZE,
                            opt_iter=5000,
                            test_size=TEST_SIZE,
                            work_dir=WORK_DIR,
                            dataset=DATA,
                            random_restart_number=RRN,
                            method=METHOD)
            
        elif DATA == 'inf_data':
            if SET_SIZE is None:
                runs = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            else:
                runs = [SET_SIZE]
            #run each size n_datasets times
            for run_num in range(N_DATASETS):
                size_id = (CORE) % len(runs)
                num_points = runs[size_id]
                tf.print("Loading inf_data with {} points...".format(num_points))
                # x, y, target = get_inf_data(func='cubic',
                #                         noise_std=2.0,
                #                         num_points=num_points)
                x, y, target = get_inf_data(func='sinc',
                                        noise_std=0.1,
                                        num_points=num_points)
                
                random_restart_number = 100 * (RRN + CORE) + run_num
                                        
                inf_data_optGPLVM.run(
                                x=x,
                                y=y,
                                target=target,
                                num_inducing=NUM_INDUCING,
                                opt_iter=N_ITERS,
                                minibatch_size=MINIBATCH_SIZE,
                                optimiser=OPTIMISER,
                                work_dir=WORK_DIR,
                                random_restart_number=random_restart_number,
                                method=METHOD)
                
        elif DATA == 'num_ind':
            runs = [25, 50, 100, 200, 400, 800, 1000]
            for run_num in range(N_DATASETS):
                size_id = (CORE) % len(runs)
                num_ind = runs[size_id]
                data_size = 1000
                tf.print("Loading inf_data with {} inducing points...".format(num_ind))
                x, y, target = get_inf_data(func='cubic',
                                        noise_std=2.0,
                                        num_points=data_size)
                
                random_restart_number = 100 * (RRN + CORE) + run_num
                                        
                inf_data_optGPLVM.run(
                                x=x,
                                y=y,
                                target=target,
                                num_inducing=num_ind,
                                opt_iter=N_ITERS,
                                minibatch_size=MINIBATCH_SIZE,
                                optimiser=OPTIMISER,
                                work_dir=WORK_DIR,
                                random_restart_number=random_restart_number,
                                method=METHOD)

if __name__ == "__main__":

    # set up argument parser
    parser = argparse.ArgumentParser(description='Run experiment')

    # set these arguments from the command line:
    parser.add_argument('--core', '-c', type=int, default=0, 
                        help='core number (only used if running on HPC). (default: 0)')
    parser.add_argument('--work_dir', '-w', type=str, default='.',
                        help='working directory')
    parser.add_argument('--data', '-d', type=str, default='an_pairs',
                        help='dataset to use')
    parser.add_argument('--n_folds', '-k', type=int, default=5,
                        help='number of folds for cross-validation')
    parser.add_argument('--n_datasets', '-nd', type=int, default=1,
                        help='number of datasets to fit per core')
    parser.add_argument('--n_iters', '-ni', type=int, default=10,
                        help='number of hyperparameter settings to try')
    parser.add_argument('--set_size', '-ss', type=int, default=None,
                        help='number of data points to use (inf_data only)')
    parser.add_argument('--test_size', '-ts', type=float, default=0.2,
                        help='proportion of data to use for testing')
    parser.add_argument('--random_restart_num', '-rrn', type=int, default=2,
                        help='random restart seed')
    parser.add_argument('--method', '-m', type=str, default='general',
                        help='method to use for fitting')
    parser.add_argument('--optimiser', '-o', type=str, default='adam',
                        help='optimiser to use for fitting')
    parser.add_argument('--num_inducing', '-in', type=int, default=100,
                        help='number of inducing points')
    parser.add_argument('--minibatch_size', '-mb', type=int, default=100,
                        help='minibatch size for optimiser')
    
    args = parser.parse_args()

    #check if all arguments are valid
    if args.n_folds < 2:
        raise ValueError("k must be greater than 1")

    if args.n_datasets < 1:
        raise ValueError("n_datasets must be greater than 0")

    if args.n_iters < 1:
        raise ValueError("n_iters must be greater than 0")

    if args.test_size < 0 or args.test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")

    if args.random_restart_num < 0:
        raise ValueError("random_restart_num must be greater than 0")

    if args.core < 0:
        raise ValueError("core must be greater than 0")
    
    if args.num_inducing < 1:
        raise ValueError("num_inducing must be greater than 0")
    
    if args.minibatch_size < 1:
        raise ValueError("minibatch_size must be greater than 0")
    
    valid_opt = ['adam', 'natgrad']
    if args.optimiser not in valid_opt:
        raise ValueError("optimiser must be one of the following: {}".format(valid_opt))

    # if args.method not in ['bayes', 'general']:
    #     raise ValueError("method must be one of the following: 'bayes', 'general'")

    #check if work_dir exists
    if not os.path.isdir(args.work_dir):
        raise ValueError("work_dir must be a valid directory")

    #check if data is one of the valid datasets
    valid_data = ['an_pairs', 'tueb', 'cha', 'inf_data', 'num_ind']
    if args.data not in valid_data:
        raise ValueError("data must be one of the following: {}".format(valid_data))

    print("Running with arguments:")
    [print("{}: {}".format(k, v)) for k, v in vars(args).items()]
    print('='*80)

    #run experiment
    run_experiment(args)
