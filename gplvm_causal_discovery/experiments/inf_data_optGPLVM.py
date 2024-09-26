"""
This method fits the generalised GPLVM to an increasing number
of datapoints from a fixed function with additive noise.
Returns the marginal likelihood and predictive loss for each
dataset size.
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

from utils import preprocess
from methods.gplvm_general import causal_scores
from methods.gplvm_bayes import causal_score_gplvm

def run(x: np.ndarray,
        y: np.ndarray,
		target: int,
		num_inducing: int=100,
		opt_iter: int=10000,
		minibatch_size: int=100,
		plot_fit: bool=True,
        work_dir: str='.',
        random_restart_number: Optional[int]=0,
		method: str='general',
		optimiser: str='adam',
    ):
	"""
	"""
	results_path = Path(work_dir) / 'results/' / 'inf_data'
	set_size = x.shape[0]

	#set random seed based on random restart
	np.random.seed(1000 * random_restart_number)
	tf.random.set_seed(1000 * random_restart_number)

	# get current dataset and preprocess
	trainval_data, _ = preprocess(x, y, test_size=0.0, shuffle_seed=random_restart_number)

	tf.print("set size: {} | num_ind: {} | target: {}".format(set_size, num_inducing, target))

	pair_results = {'size': set_size,
				 	'num_inducing': num_inducing,
					'minibatch_size': minibatch_size,
					'optimiser': optimiser,
					'method': method,
					'target': target,
					'causal_marginal_likelihood': None,
					'causal_predictive_loss': None,
					'anti_marginal_likelihood': None,
					'anti_predictive_loss': None}
        
	#do the following for both causal directions
	for causal in [True, False]:
		if causal:
			tf.print("Causal direction")
			#do nothing (leave data as is)
			trainval_data = (trainval_data[0], trainval_data[1])
		else:
			tf.print("Anti-causal direction")
			#swap x and y
			trainval_data = (trainval_data[1], trainval_data[0])

		#get best marginal likelihood and predictive loss for this dataset size
		tf.print("Fitting model...")
		if method == 'bayes':
			marg_lls, pred_losses = causal_score_gplvm(trainval_data,
													num_inducing=num_inducing,
													opt_iter=opt_iter,
													minibatch_size=minibatch_size,
													optimiser=optimiser,
													plot_fit=plot_fit,
													causal=causal,
													set_size=set_size,
													random_restart_number=random_restart_number,
													dataset_name='inf_data')
		elif method == 'general':
			marg_lls, pred_losses = causal_scores(trainval_data,
														num_inducing=num_inducing,
														opt_iter=opt_iter,
														minibatch_size=minibatch_size,
														optimiser=optimiser,
														plot_fit=plot_fit,
														causal=causal,
														set_size=set_size,
														random_restart_number=random_restart_number,
														dataset_name='inf_data')

		
		# sum marg_ll and predictive losses from marginal and conditional models
		total_marg_ll = marg_lls[0] + marg_lls[1]
		total_pred_loss = pred_losses[0] + pred_losses[1]

		#save results
		if causal:
			pair_results['causal_marginal_likelihood'] = total_marg_ll
			pair_results['causal_predictive_loss'] = total_pred_loss
		else:
			pair_results['anti_marginal_likelihood'] = total_marg_ll
			pair_results['anti_predictive_loss'] = total_pred_loss
			
		
	#save results to csv
	new_row = pd.DataFrame(pair_results, index=[0])
	fname = results_path / 'results.csv'
	new_row.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

	tf.print("Finished with size {}".format(set_size))







		
            
        

        



        
        

        
    
