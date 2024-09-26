import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from sklearn.preprocessing import StandardScaler
import os

from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from models.BayesGPLVM import BayesianGPLVM
from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from models.GeneralisedGPLVM import GeneralisedGPLVM
from models.GeneralisedUnsupGPLVM import GeneralisedUnsupGPLVM

class Search:
    def __init__(self,
                 work_dir: Optional[str]=None,
                 ):
        '''
        Args:
            work_dir: directory to save results
        '''
        self.work_dir = work_dir if work_dir is not None else Path(__file__).parent.parent.absolute()

    def crossValidate(self,
                      data,
                      hyperparams,
                      num_inducing,
                      method: str,
                      opt_iter: int=1000,
                      num_minibatch: Optional[int]=100,
                      causal: Optional[bool]=None,
                      pair_index: Optional[int]=0,
                      iteration: Optional[int]=0,
                      random_restart_number: Optional[int]=0,
                      plot_fit: Optional[bool]=False
                      ):
        '''
        Perform k-fold cross validation

        Args:
            x: input data including validation
            y: target data including validation
            k: number of folds (int)
            model: GPflow model

        Returns:
            losses: mean squared error
        '''

        #get hyperparameters
        k = hyperparams['k'].astype(int)

        #get folds
        folds = self.getFolds(data, k)

        fold_losses = pd.DataFrame(columns=['train_loss', 'val_loss']) #df to store losses for each fold

        for i in range(k):
            tf.print("Fold: {}/{}".format(i+1, k))

            #get train and validation data for fold
            train_data_norm, val_data_norm = self.getTrainVal(folds, i)

            jitter = 1e-4
            while True:
                try:
                    if method == 'general':
                        #train model with generalised GPLVM
                        trained_model = self.train(
                            data=train_data_norm,
                            num_inducing=num_inducing,
                            hyperparams=hyperparams,
                            num_minibatch=num_minibatch,
                            opt_iter=opt_iter,
                            adam_lr=0.05,
                            causal=causal,
                            pair_index=pair_index,
                            iteration=iteration,
                            fold=i,
                            random_restart_number=random_restart_number,
                            jitter=jitter,
                            plot_fit=plot_fit,
                        )
                    
                    elif method == 'bayes':
                        #train model with Bayesian GPLVM (no minibatching)
                        trained_model = self.train_Bayes(
                            data=train_data_norm,
                            num_inducing=num_inducing,
                            hyperparams=hyperparams,
                            causal=causal,
                            pair_index=pair_index,
                            iteration=iteration,
                            fold=i,
                            random_restart_number=random_restart_number,
                            jitter=jitter,
                            plot_fit=plot_fit,
                        )

                    break
                #if we get LB is not finite error, increase jitter
                except tf.errors.InvalidArgumentError as exc:
                    jitter *= 10
                    tf.print(f"WARNING: Cholesky decomposition failed, increasing jitter to {jitter}")
            
            #get train and validation loss
            train_data_idx = np.arange(train_data_norm[0].shape[0])
            val_data_idx = np.arange(val_data_norm[0].shape[0])
            if method == 'general':
                if self.isMarginal:
                    train_loss = -trained_model.predictive_score((train_data_norm[0], train_data_idx))
                    val_loss = -trained_model.predictive_score((val_data_norm[0], val_data_idx))
                else:
                    train_loss = -trained_model.predictive_score((train_data_norm[0], train_data_norm[1], train_data_idx))
                    val_loss = -trained_model.predictive_score((val_data_norm[0], val_data_norm[1], val_data_idx))

            elif method == 'bayes':
                if self.isMarginal:
                    train_loss = -trained_model.predictive_score(train_data_norm[0])
                    val_loss = -trained_model.predictive_score(val_data_norm[0])
                else:
                    train_loss = -trained_model.predictive_score((train_data_norm[0], train_data_norm[1]))
                    val_loss = -trained_model.predictive_score((val_data_norm[0], val_data_norm[1]))

            
            #if either is -np.inf, print warning
            if train_loss == np.inf or val_loss == np.inf:
                tf.print("WARNING: np.inf loss")


            #store losses
            fold_losses.loc[len(fold_losses)] = [train_loss, val_loss]
            
        #take mean of each loss
        train_loss_mean = fold_losses['train_loss'].mean().astype(np.float64)
        val_loss_mean = fold_losses['val_loss'].mean().astype(np.float64)

        return train_loss_mean, val_loss_mean

    def randomSearch(self, 
                     trainval_data: np.ndarray,
                     test_data: np.ndarray,
                     method: str,
                     hyperspace: dict,
                     num_inducing: int,
                     n_iter: int=10,
                     opt_iter: int=1000,
                     num_minibatch: Optional[int]=100,
                     causal: Optional[bool]=None,
                     pair_index: int=0,
                     random_restart_number: Optional[int]=0,
                     plot_fit: bool=False,
                     dataset_name: str=''
                     ):
        '''
        Args:
            x_raw: raw data
            y_raw: raw data
            n_iter: number of iterations to run
            dist: distribution to sample from
        '''
        self.dataset_name = dataset_name
        self.method = method

        fname = self.work_dir / 'results' / self.dataset_name / 'loss_table_core{}.csv'.format(pair_index - 1)

        # infer model type
        model_type = 'marginal' if self.isMarginal else 'conditional'
        
        #run cross validation with different hyperparameters
        loss_table = pd.DataFrame({'dataset': pd.Series([], dtype='str'),
                                'model': pd.Series([], dtype='str'),
                                'causal': pd.Series([], dtype='bool'),
                                'k': pd.Series([], dtype='int'),
                                'sq_exp_var': pd.Series([], dtype='float64'),
                                'sq_exp_len_obs': pd.Series([], dtype='float64'),
                                'sq_exp_len_lat': pd.Series([], dtype='float64'),
                                'lin_var': pd.Series([], dtype='float64'),
                                'mat_var': pd.Series([], dtype='float64'),
                                'mat_len_obs': pd.Series([], dtype='float64'),
                                'mat_len_lat': pd.Series([], dtype='float64'),
                                'rquad_var': pd.Series([], dtype='float64'),
                                'rquad_len_obs': pd.Series([], dtype='float64'),
                                'rquad_len_lat': pd.Series([], dtype='float64'),
                                'likelihood_variance': pd.Series([], dtype='float64'),
                                'train_loss': pd.Series([], dtype='float64'),
                                'val_loss': pd.Series([], dtype='float64'),
                                'random_restart_number': pd.Series([], dtype='int')
                                })
        
        samples = pd.DataFrame(columns=hyperspace.keys())

        #sample from distribution
        for hyperparam in hyperspace.keys():
            #draw n_iter samples from distribution
            if hyperparam == 'k':
                samples[hyperparam] = hyperspace[hyperparam].rvs(n_iter).astype(int)

            else:
                samples[hyperparam] = hyperspace[hyperparam].rvs(n_iter)

            # else:
            #     for i in range(n_iter):
            #         samples[hyperparam] = pd.Series([np.zeros(x_dim) for i in range(n_iter)])
            #         samples[hyperparam].iloc[i] = hyperspace[hyperparam].rvs(x_dim)
            
        
        for i in range(n_iter):

            tf.print("Iteration: {}/{}".format(i+1, n_iter))

            #ith set of hyperparameters
            hyperparams = samples.iloc[i]

            #cross validate
            train_loss, val_loss = self.crossValidate(trainval_data,
                                                      hyperparams,
                                                      num_inducing,
                                                      method=method,
                                                      opt_iter=opt_iter,
                                                      num_minibatch=num_minibatch,
                                                      causal=causal,
                                                      pair_index=pair_index,
                                                      iteration=i,
                                                      random_restart_number=random_restart_number,
                                                      plot_fit=False
                                                    )

            # add new row to loss_table DataFrame
            new_row = pd.DataFrame({
                'dataset': pair_index,
                'model': model_type,
                'causal': causal,
                'k': hyperparams['k'],
                'sq_exp_var': hyperparams['sq_exp_var'],
                'sq_exp_len_obs': hyperparams['sq_exp_len_obs'],
                'sq_exp_len_lat': hyperparams['sq_exp_len_lat'],
                'lin_var': hyperparams['lin_var'],
                'mat_var': hyperparams['mat_var'],
                'mat_len_obs': hyperparams['mat_len_obs'],
                'mat_len_lat': hyperparams['mat_len_lat'],
                'rquad_var': hyperparams['rquad_var'],
                'rquad_len_obs': hyperparams['rquad_len_obs'],
                'rquad_len_lat': hyperparams['rquad_len_lat'],
                'likelihood_variance': hyperparams['likelihood_variance'],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'random_restart_number': random_restart_number
            }, index=[0])
            loss_table = pd.concat([loss_table, new_row], ignore_index=True)

            #add new row to csv (create csv if it doesn't exist)
            new_row.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)
            
        test_loss = self.getBestLoss(loss_table=loss_table,
                                     trainval_data=trainval_data,
                                     test_data=test_data,
                                     num_inducing=num_inducing,
                                     opt_iter=opt_iter,
                                     num_minibatch=num_minibatch,
                                     causal=causal,
                                     pair_index=pair_index,
                                     random_restart_number=random_restart_number,
                                     iteration=n_iter,
                                     plot_fit=plot_fit
                                     )

        return test_loss
        

    def getBestLoss(self, 
                    loss_table: pd.DataFrame,
                    trainval_data: np.ndarray,
                    test_data: np.ndarray,
                    num_inducing: int,
                    opt_iter: int=1000,
                    num_minibatch: Optional[int]=100,
                    causal: Optional[bool]=None,
                    pair_index: int=0,
                    random_restart_number: Optional[int]=0,
                    iteration: Optional[int]=0,
                    plot_fit: bool=False
                    ):

        '''
        Args:
            loss_table: DataFrame containing losses for each set of hyperparameters
        Returns:
            best_hyperparams: hyperparameters with lowest validation loss
        '''

        #get best hyperparameters based on validation loss
        best_hyperparams = loss_table.iloc[loss_table['val_loss'].astype(float).idxmin()]
        # exclude dataset, model, causal, k, train_loss, val_loss, random_restart_number
        best_hyperparams = best_hyperparams.drop(['dataset', 
                                                  'model', 
                                                  'causal', 
                                                  'k', 
                                                  'train_loss', 
                                                  'val_loss', 
                                                  'random_restart_number'])

        #train on all trainval data with best hyperparameters
        tf.print("Training with best hyperparams on trainval data for test loss")
        if self.method == 'general':
            trainval_model = self.train(
                data=trainval_data,
                num_inducing=num_inducing,
                hyperparams=best_hyperparams,
                num_minibatch=num_minibatch,
                opt_iter=opt_iter,
                adam_lr=0.05,
                causal=causal,
                pair_index=pair_index,
                iteration=iteration,
                random_restart_number=random_restart_number,
                plot_fit=plot_fit
                )
            
        elif self.method == 'bayes':
            trainval_model = self.train_Bayes(
                data=trainval_data,
                num_inducing=num_inducing,
                hyperparams=best_hyperparams,
                adam_lr=0.05,
                causal=causal,
                pair_index=pair_index,
                iteration=iteration,
                random_restart_number=random_restart_number,
                plot_fit=plot_fit
                )
                                    
        #evaluate on test data
        test_data_idx = np.arange(test_data[0].shape[0])
        if self.method == 'general':
            if self.isMarginal:
                test_loss = -trainval_model.predictive_score((test_data[0], test_data_idx))
            else:
                test_loss = -trainval_model.predictive_score((test_data[0], test_data[1], test_data_idx))
        elif self.method == 'bayes':
            if self.isMarginal:
                test_loss = -trainval_model.predictive_score(test_data[0])
            else:
                test_loss = -trainval_model.predictive_score((test_data[0], test_data[1]))


        return test_loss.numpy()
    
    def run_optimizer(
        self,
        model: GeneralisedGPLVM,
        train_dataset: tf.data.Dataset,
        iterations: int,
        data_size: int,
        minibatch_size: int,
        adam_lr: float,
    ):
        """
        Utility function running the Adam optimizer
        Modified from https://gpflow.readthedocs.io/en/master/notebooks/advanced/gps_for_big_data.html
        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam(adam_lr)

        # Do not train kernel and likelihood, determined by CV
        gpflow.utilities.set_trainable(model.kernel, False)
        gpflow.utilities.set_trainable(model.likelihood, False)

        # Train latent variables
        gpflow.utilities.set_trainable(model.X_data_mean, True)
        gpflow.utilities.set_trainable(model.X_data_var, True)
        gpflow.utilities.set_trainable(model.inducing_variable, True)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

        iterator = range(iterations)
        for step in iterator:
            optimization_step()
            neg_elbo = training_loss().numpy()
            logf.append(neg_elbo)
            # It is possible to include a stopping criteria here. However there
            # is a risk that the training will be stopped too soon and the
            # ELBO achieved will not be as high as it could be
        return logf

class Marginal(Search):
    def __init__(self):
        super().__init__()
        self.isMarginal = True

    def getFolds(self, data, k):
        '''
        Split data into k folds when tuple with shape ([n, 1])
        '''

        #split data into k folds throwing away remainder
        n_pts = data[0].shape[0]
        remainder = n_pts % k
        folds = np.array_split(data[0][:n_pts-remainder], k, axis=0)

        print("folds: ", folds)


        #return an array of tuples
        return [(fold, ) for fold in folds]

    def getTrainVal(self, folds, i):
        '''
        Get train and validation data for fold i
        '''

        val_data = folds[i]
        #everything else is train data, so concatenate all other folds into a tuple of one array
        train_data = np.concatenate([fold for j, fold in enumerate(folds) if j != i], axis=0)

        print("train_data shape: ", train_data.shape)
        print("val_data shape: ", val_data.shape)

        #normalise data (fit scaler to train data, then transform train and val data)
        scaler_train = StandardScaler()
        train_data_norm = (scaler_train.fit_transform(train_data[0]), )
        val_data_norm = (scaler_train.transform(val_data[0]), )

        return train_data_norm, val_data_norm

    def train_Bayes(
        self,
        data: np.ndarray,
        num_inducing: int,
        hyperparams: dict,
        causal: Optional[bool] = None,
        iteration: Optional[int] = 0,
        fold: Optional[int] = 0,
        adam_lr: Optional[float] = 0.05,
        random_restart_number: Optional[int] = 0,
        pair_index: Optional[int] = 0,
        jitter: Optional[float] = 1e-4,
        plot_fit: Optional[bool] = False
        ):

        causal_offset = 0 if causal else 1
        marg_cond_offset = 0 # 0 if marginal, 1 if conditional
        jitter_offset = -np.log10(jitter)
        rng = np.random.default_rng(seed=int(pair_index * 1e8
                                             + causal_offset * 1e7
                                             + marg_cond_offset * 2e7
                                             + jitter_offset * 1e6
                                             + iteration * 1e4
                                             + fold * 1e2
                                             + random_restart_number))
        # extract hyperparameters
        sq_exp_variance = hyperparams['sq_exp_var']
        kernel_lengthscale = hyperparams['sq_exp_len_obs']
        likelihood_variance = hyperparams['likelihood_variance']
        lin_variance = hyperparams['lin_var']

        tf.print("Trying: sqe_var: %.5f, len_obs: %.5f, lin_var: %.5f, likelihood_var: %.5f"%(
            sq_exp_variance, kernel_lengthscale, lin_variance, likelihood_variance
            ))

        # extract data
        y = data[0]

        latent_dim = 1

        # Define kernel
        sq_exp = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale], variance=sq_exp_variance)
        linear_kernel = gpflow.kernels.Linear(variance=lin_variance)
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])

        # Initialise approx posterior and prior
        X_mean_init = 0.1 * tf.cast(y, default_float())
        X_var_init = tf.cast(
            rng.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
        )
        x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
        inducing_variable = gpflow.inducing_variables.InducingPoints(
            rng.normal(size=(num_inducing, latent_dim)),
        )

        # Define marginal model
        marginal_model = BayesianGPLVM(
            data=y,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            X_prior_var=x_prior_var,
            jitter=jitter,
            inducing_variable=inducing_variable
        )
        marginal_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(1e-6)
        )

        loss_fn = marginal_model.training_loss_closure()

        tf.print("Adam opt: latent hyperparameters for marginal model")
        # Do not train kernel and likelihood, determined by CV
        gpflow.utilities.set_trainable(marginal_model.kernel, False)
        gpflow.utilities.set_trainable(marginal_model.likelihood, False)

        # Train latent variables
        gpflow.utilities.set_trainable(marginal_model.X_data_mean , True)
        gpflow.utilities.set_trainable(marginal_model.X_data_var, True)
        gpflow.utilities.set_trainable(marginal_model.inducing_variable, True)

        adam_vars = marginal_model.trainable_variables
        adam_opt = tf.optimizers.Adam(adam_lr)

        @tf.function
        def optimisation_step():
            adam_opt.minimize(loss_fn, adam_vars)

        epochs = int(100)

        for epoch in range(epochs):
            optimisation_step()
        
        # Train latent variables after Adam
        tf.print("Scipy opt: latent hyperparameters for marginal model")

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            marginal_model.training_loss,
            marginal_model.trainable_variables,
            options=dict(maxiter=100000),
        )
        #tf.print("ELBO:", - marginal_model.elbo())

        if plot_fit:
            # Plot the fit to see if everything is ok
            obs_new = np.linspace(-5, 5, 1000)[:, None]

            pred_y_mean, pred_y_var = marginal_model.predict_y(
                Xnew=obs_new,
            )
            textstr = 'len_lat=%.3f\nsqe_var=%.3f\nlin_var=%.3f\nlike_var=%.3f' % (
                marginal_model.kernel.kernels[0].lengthscales.numpy(),
                marginal_model.kernel.kernels[0].variance.numpy(),
                marginal_model.kernel.kernels[1].variance.numpy(),
                marginal_model.likelihood.variance.numpy()
            )
            plt.text(-8.5, 0, textstr, fontsize=8)
            plt.fill_between(
                obs_new[:, 0],
                (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0],
                (pred_y_mean - 2 * np.sqrt(pred_y_var))[:, 0],
                alpha=0.25,
                color='C0',
                label='prediction variance'
            )
            plt.scatter(marginal_model.X_data_mean, y, c='C2', marker='.', label='training data')
            plt.plot(obs_new, pred_y_mean, c='C0', alpha=0.5, label='prediction mean')
            plt.legend()
            save_name = "marginal"
            save_dir = Path(f"{self.work_dir}/figs/run_plots/{self.dataset_name}/")
            save_dir.mkdir(
                parents=True, exist_ok=True
            )
            plt.subplots_adjust(left=0.25)
            causal_direction = "causal" if causal else "anticausal"
            plt.savefig(
                save_dir / f"pair{pair_index}_{causal_direction}_{save_name}.jpg",
            )
            plt.close()
        else:
            pass
        return marginal_model

    def train(
        self,
        data: np.ndarray,
        hyperparams: dict,
        num_inducing: int,
        num_minibatch: int,
        opt_iter: int,
        adam_lr: float,
        causal: Optional[bool] = None,
        iteration: Optional[int] = 0,
        fold: Optional[int] = 0,
        random_restart_number: Optional[int] = 0,
        pair_index: Optional[int] = 0,
        jitter: Optional[float] = 1e-4,
        plot_fit: Optional[bool] = False
    ):
        #set random seed
        causal_offset = 0 if causal else 1
        marg_cond_offset = 0 # 0 if marginal, 1 if conditional
        jitter_offset = -np.log10(jitter)
        rng = np.random.default_rng(seed=int(pair_index * 1e8
                                             + causal_offset * 1e7
                                             + marg_cond_offset * 2e7
                                             + jitter_offset * 1e6
                                             + iteration * 1e4
                                             + fold * 1e2
                                             + random_restart_number))
        
        # extract hyperparameters
        #

        tf.print("Trying: ", end='')
        for key in hyperparams.keys():
            tf.print("%s: %.5f, "%(key, hyperparams[key]), end='')
        tf.print("")

        # extract data
        y = data[0]

        # Define kernels
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=hyperparams['sq_exp_len_obs'],
        )
        sq_exp.variance.assign(hyperparams['sq_exp_var'])
        matern = gpflow.kernels.Matern32(lengthscales=hyperparams['mat_len_obs'])
        matern.variance.assign(hyperparams['mat_var'])
        rquadratic = gpflow.kernels.RationalQuadratic(
            lengthscales=hyperparams['rquad_len_obs'],
        )
        rquadratic.variance.assign(hyperparams['rquad_var'])
        linear_kernel = gpflow.kernels.Linear(variance=hyperparams['lin_var'])
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern, rquadratic])
        Z = np.random.randn(num_inducing, 1)

        # Define the approx posteroir
        X_mean_init = 0.1 * tf.cast(y, default_float())
        X_var_init = tf.cast(
            np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
        )

        # Define marginal model
        marginal_model = GeneralisedUnsupGPLVM(
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(variance=hyperparams['likelihood_variance']),
            num_mc_samples=1,
            inducing_variable=Z,
            batch_size=num_minibatch,
        )
        # Run optimisation
        data_idx = np.arange(y.shape[0])
        train_dataset = tf.data.Dataset.from_tensor_slices((y, data_idx)).repeat()

        logf = self.run_optimizer(
            model=marginal_model,
            train_dataset=train_dataset,
            iterations=opt_iter,
            adam_lr=adam_lr,
            data_size=y.shape[0],
            minibatch_size=num_minibatch,
        )

        #marginal_model.num_mc_samples = 100

        if plot_fit:
            # Plot the fit to see if everything is ok
            obs_new = np.linspace(-5, 5, 1000)[:, None]

            pred_y_mean, pred_y_var = marginal_model.predict_y(
                Xnew=obs_new,
            )
            text_sqe = 'sqe_len=%.3f\nsqe_var=%.3f' % (
                marginal_model.kernel.kernels[0].lengthscales.numpy(),
                marginal_model.kernel.kernels[0].variance.numpy(),
            )
            text_lin = 'lin_var=%.3f' % (
                marginal_model.kernel.kernels[1].variance.numpy(),
            )
            text_matern = 'matern_len=%.3f\nmatern_var=%.3f' % (
                marginal_model.kernel.kernels[2].lengthscales.numpy(),
                marginal_model.kernel.kernels[2].variance.numpy(),
            )
            text_rquad = 'rquad_len=%.3f\nrquad_var=%.3f' % (
                marginal_model.kernel.kernels[3].lengthscales.numpy(),
                marginal_model.kernel.kernels[3].variance.numpy(),
            )
            text_likelihood = 'likelihood_var=%.3f' % (
                marginal_model.likelihood.variance.numpy(),
            )
            textstr = text_sqe + '\n' + text_lin + '\n' + text_matern + '\n' + text_rquad + '\n' + text_likelihood

            plt.text(-8.5, 0, textstr, fontsize=8)
            plt.fill_between(
                obs_new[:, 0],
                (pred_y_mean + 2 * np.sqrt(pred_y_var))[:, 0],
                (pred_y_mean - 2 * np.sqrt(pred_y_var))[:, 0],
                alpha=0.25,
                color='C0',
                label='prediction variance'
            )
            plt.scatter(marginal_model.X_data_mean, y, c='C2', marker='.', label='training data')
            plt.plot(obs_new, pred_y_mean, c='C0', alpha=0.5, label='prediction mean')
            plt.legend()
            save_name = "marginal"
            save_dir = Path(f"{self.work_dir}/figs/run_plots/{self.dataset_name}/")
            save_dir.mkdir(
                parents=True, exist_ok=True
            )
            plt.subplots_adjust(left=0.25)
            causal_direction = "causal" if causal else "anticausal"
            plt.savefig(
                save_dir / f"pair{pair_index}_{causal_direction}_{save_name}.jpg",
            )
            plt.close()
        else:
            pass
        return marginal_model

class Conditional(Search):
    def __init__(self):
        super().__init__()
        self.isMarginal = False

    def getFolds(self, data, k):
        '''
        Split data into k folds when tuple with shape ([n, 1], [n, 1])
        '''

        #split data into k equal folds, throw away remainder
        n_pts = data[0].shape[0]
        remainder = n_pts % k
        folds_x = np.array_split(data[0][:n_pts - remainder], k, axis=0)
        folds_y = np.array_split(data[1][:n_pts - remainder], k, axis=0)

        #return an array of tuples
        return [(fold_x, fold_y) for fold_x, fold_y in zip(folds_x, folds_y)]
    
    def getTrainVal(self, folds, i):
        '''
        Get train and validation data for fold i
        '''

        val_data = folds[i]
        #everything else is train data, so concatenate all other folds into a tuple of two arrays
        train_data_x = np.concatenate([fold[0] for j, fold in enumerate(folds) if j != i], axis=0)
        train_data_y = np.concatenate([fold[1] for j, fold in enumerate(folds) if j != i], axis=0)
        train_data = (train_data_x, train_data_y)

        #normalise data (fit scaler to train data, then transform train and val data)
        scaler_train_x = StandardScaler()
        scaler_train_y = StandardScaler()
        train_data_norm = (scaler_train_x.fit_transform(train_data[0]), scaler_train_y.fit_transform(train_data[1]))
        val_data_norm = (scaler_train_x.transform(val_data[0]), scaler_train_y.transform(val_data[1]))

        return train_data_norm, val_data_norm

    def train_Bayes(
        self,
        data: np.ndarray,
        num_inducing: int,
        hyperparams: dict,
        causal: Optional[bool] = None,
        iteration: Optional[int] = 0,
        fold: Optional[int] = 0,
        random_restart_number: Optional[int] = 0,
        adam_lr: Optional[float] = 0.05,
        jitter: Optional[float] = 1e-4,
        pair_index: Optional[int] = 0,
        plot_fit: Optional[bool] = False,
        ):
        """
        Train a conditional model using a partially observed GPLVM.
        """
        
        causal_offset = 0 if causal else 1
        marg_cond_offset = 1 # 0 for marginal, 1 for conditional
        jitter_offset = -np.log10(jitter)
        rng = np.random.default_rng(seed=int(pair_index * 1e8
                                             + causal_offset * 1e7
                                             + marg_cond_offset * 2e7
                                             + jitter_offset * 1e6
                                             + iteration * 1e4
                                             + fold * 1e2
                                             + random_restart_number))
        # unpack data
        x, y = data[0], data[1]

        #extract hyperparams
        kernel_len_obs = hyperparams["sq_exp_len_obs"]
        kernel_len_lat = hyperparams["sq_exp_len_lat"]
        sq_exp_var = hyperparams["sq_exp_var"]
        likelihood_variance = hyperparams["likelihood_variance"]
        lin_var = hyperparams["lin_var"]


        #print hyperparams
        tf.print("Trying: len_obs=%.5f, len_lat=%.5f, sq_exp_var=%.5f, lin_var=%.5f, likelihood_variance=%.5f"%(
            kernel_len_obs, kernel_len_lat, sq_exp_var, lin_var, likelihood_variance
        ))


        latent_dim = 1

        # Define hyperparameters
        X_mean_init = 0.1 * tf.cast(y, default_float())
        sq_exp = gpflow.kernels.SquaredExponential(lengthscales=[kernel_len_obs] + [kernel_len_lat])
        sq_exp.variance.assign(sq_exp_var + 1e-20)
        linear_kernel = gpflow.kernels.Linear(variance=lin_var + 1e-20)
        X_var_init = tf.cast(
            rng.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
        )
        x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
        inducing_variable = gpflow.inducing_variables.InducingPoints(
            np.concatenate(
                [
                    np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                    rng.normal(size=(num_inducing, 1)),
                ],
                axis=1
            )
        )


        # Define conditional model
        conditional_model = PartObsBayesianGPLVM(
            data=y,
            in_data=x,
            kernel=kernel,
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            X_prior_var=x_prior_var,
            jitter=jitter,
            inducing_variable=inducing_variable,
        )

        conditional_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(lower=1e-6)
        )

        loss_fn = conditional_model.training_loss_closure()

        tf.print("Adam opt: latent hyperparameters for conditional model")
        # Do not train kernel and likelihood, determined by CV
        gpflow.utilities.set_trainable(conditional_model.kernel, False)
        gpflow.utilities.set_trainable(conditional_model.likelihood, False)

        # Train latent variables with Adam first
        gpflow.utilities.set_trainable(conditional_model.X_data_mean, True)
        gpflow.utilities.set_trainable(conditional_model.X_data_var, True)
        gpflow.utilities.set_trainable(conditional_model.inducing_variable, True)

        adam_vars = conditional_model.trainable_variables
        adam_opt = tf.optimizers.Adam(adam_lr)

        @tf.function
        def optimisation_step():
            adam_opt.minimize(loss_fn, adam_vars)

        epochs = int(100)

        for epoch in range(epochs):
            optimisation_step()

        # Train latent variables after Adam
        tf.print("Scipy opt: latent hyperparameters for conditional model")

        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            conditional_model.training_loss,
            conditional_model.trainable_variables,
            options=dict(maxiter=100000),
        )
        #tf.print("ELBO:", - conditional_model.elbo())

        if plot_fit:
            # Plot the fit to see if everything is ok
            num_lat = 50
            num_gp = 50
            Xnew = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
            samples = conditional_model.predict_full_samples_layer(Xnew, 
                                                         obs_noise=True, 
                                                         num_latent_samples=num_lat,
                                                         num_gp_samples=num_gp)
            
            samples = samples.numpy().reshape(num_gp, num_lat, -1)
            
            Xnew = np.tile(Xnew, (num_gp, num_lat, 1))
            
            plt.scatter(Xnew, samples, color="C0", alpha=0.01, label="samples", marker=".", edgecolors="none", s=200)
            plt.scatter(x, y, color="C2", label="data", marker=".", alpha=0.5, edgecolors="none", s=100)

            textstr = 'kern_len_obs=%.3f\nkern_len_lat=%.3f\nsqe_var=%.3f\nlin_var=%.3f\nlik_var=%.3f' % (
                conditional_model.kernel.kernels[0].lengthscales.numpy()[0],
                conditional_model.kernel.kernels[0].lengthscales.numpy()[1],
                conditional_model.kernel.kernels[0].variance.numpy(),
                conditional_model.kernel.kernels[1].variance.numpy(),
                conditional_model.likelihood.variance.numpy(),
            )
            #put text on plot
            plt.text(-8.5, 0, textstr, fontsize=8)

            #legend should have alpha 1
            leg = plt.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            save_name = "conditional"
            save_dir = Path(f"{self.work_dir}/figs/run_plots/{self.dataset_name}/")
            save_dir.mkdir(
                parents=True, exist_ok=True
            )
            plt.subplots_adjust(left=0.25) 
            causal_direction = "causal" if causal else "anticausal"
            plt.savefig(
                save_dir / f"pair{pair_index}_{causal_direction}_{save_name}.jpg",
            )
            plt.close()
        else:
            pass

        return conditional_model
    
    def train(
        self,
        data: np.ndarray,
        hyperparams: dict,
        num_inducing: int,
        num_minibatch: int,
        opt_iter: int,
        adam_lr: float,
        causal: Optional[bool] = None,
        iteration: Optional[int] = 0,
        fold: Optional[int] = 0,
        random_restart_number: Optional[int] = 0,
        pair_index: Optional[int] = 0,
        jitter: Optional[float] = 1e-4,
        plot_fit: Optional[bool] = False
    ):
        #set random seed
        causal_offset = 0 if causal else 1
        marg_cond_offset = 1 # 0 if marginal, 1 if conditional
        jitter_offset = -np.log10(jitter)
        rng = np.random.default_rng(seed=int(pair_index * 1e8
                                             + causal_offset * 1e7
                                             + marg_cond_offset * 2e7
                                             + jitter_offset * 1e6
                                             + iteration * 1e4
                                             + fold * 1e2
                                             + random_restart_number))
        
        # extract hyperparameters
        # sq_exp_var = hyperparams["sq_exp_var"]
        # sq_exp_len_obs = hyperparams["sq_exp_len_obs"]
        # sq_exp_len_lat = hyperparams["sq_exp_len_lat"]

        # mat_var = hyperparams["mat_var"]
        # mat_len_obs = hyperparams["mat_len_obs"]
        # mat_len_lat = hyperparams["mat_len_lat"]

        # rquad_var = hyperparams["rquad_var"]
        # rquad_len_obs = hyperparams["rquad_len_obs"]
        # rquad_len_lat = hyperparams["rquad_len_lat"]

        # lin_var = hyperparams["lin_var"]

        # likelihood_variance = hyperparams["likelihood_variance"]

        tf.print("Trying: ", end='')
        for key in hyperparams.keys():
            tf.print("%s: %.5f, "%(key, hyperparams[key]), end='')
        tf.print("")

        # extract data
        x, y = data[0], data[1]

        # Define kernels
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[hyperparams["sq_exp_len_obs"], hyperparams["sq_exp_len_lat"]]
        )
        sq_exp.variance.assign(hyperparams["sq_exp_var"])
        matern = gpflow.kernels.Matern32(
            lengthscales=[hyperparams["mat_len_obs"], hyperparams["mat_len_lat"]]
        )
        matern.variance.assign(hyperparams["mat_var"])
        rquadratic = gpflow.kernels.RationalQuadratic(
            lengthscales=[hyperparams["rquad_len_obs"], hyperparams["rquad_len_lat"]]
        )
        rquadratic.variance.assign(hyperparams["rquad_var"])
        linear_kernel = gpflow.kernels.Linear(variance=hyperparams["lin_var"])
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel, matern, rquadratic])

        Z = np.concatenate(
            [
                np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
            ],
            axis=1,
        )

        # Define the approx posteroir
        X_mean_init = 0.01 * tf.cast(y, default_float())
        X_var_init = tf.cast(
            np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
        )

        # Define the conditional model
        conditional_model = GeneralisedGPLVM(
            X_data_mean=X_mean_init,
            X_data_var=X_var_init,
            kernel=kernel,
            likelihood=gpflow.likelihoods.Gaussian(variance=hyperparams["likelihood_variance"]),
            num_mc_samples=1,
            inducing_variable=Z,
            batch_size=num_minibatch,
        )

        # Run optimisation
        data_idx = np.arange(y.shape[0])
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x, y, data_idx)
        ).repeat()
        logf = self.run_optimizer(
            model=conditional_model,
            train_dataset=train_dataset,
            iterations=opt_iter,
            adam_lr=adam_lr,
            data_size=y.shape[0],
            minibatch_size=num_minibatch,
        )

        #conditional_model.num_mc_samples = 100

        if plot_fit:
            # Plot the fit to see if everything is ok
            num_lat = 50
            num_gp = 50
            Xnew = np.linspace(x.min(), x.max(), 50).reshape(-1, 1)
            samples = conditional_model.predict_full_samples_layer(Xnew, 
                                                         obs_noise=True, 
                                                         num_latent_samples=num_lat,
                                                         num_gp_samples=num_gp)
            
            samples = samples.numpy().reshape(num_gp, num_lat, -1)
            
            Xnew = np.tile(Xnew, (num_gp, num_lat, 1))
            
            plt.scatter(Xnew, samples, color="C0", alpha=0.01, label="samples", marker=".", edgecolors="none", s=200)
            plt.scatter(x, y, color="C2", label="data", marker=".", alpha=0.5, edgecolors="none", s=100)

            text_sqe = 'sqe: len_obs=%.3f, len_lat=%.3f, var=%.3f' % (
                conditional_model.kernel.kernels[0].lengthscales.numpy()[0],
                conditional_model.kernel.kernels[0].lengthscales.numpy()[1],
                conditional_model.kernel.kernels[0].variance.numpy(),
            )
            text_lin = 'lin: var=%.3f' % (
                conditional_model.kernel.kernels[1].variance.numpy(),
            )
            text_mat = 'mat: len_obs=%.3f, len_lat=%.3f, var=%.3f' % (
                conditional_model.kernel.kernels[2].lengthscales.numpy()[0],
                conditional_model.kernel.kernels[2].lengthscales.numpy()[1],
                conditional_model.kernel.kernels[2].variance.numpy(),
            )
            text_rquad = 'rquad: len_obs=%.3f, len_lat=%.3f, var=%.3f' % (
                conditional_model.kernel.kernels[3].lengthscales.numpy()[0],
                conditional_model.kernel.kernels[3].lengthscales.numpy()[1],
                conditional_model.kernel.kernels[3].variance.numpy(),
            )
            text_likelihood = 'likelihood: var=%.3f' % (
                conditional_model.likelihood.variance.numpy(),
            )
            textstr = text_sqe + "\n" + text_lin + "\n" + text_mat + "\n" + text_rquad + "\n" + text_likelihood

            #put text on plot
            plt.text(-8.5, 0, textstr, fontsize=8)

            #legend should have alpha 1
            leg = plt.legend()
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            save_name = "conditional"
            save_dir = Path(f"{self.work_dir}/figs/run_plots/{self.dataset_name}/")
            save_dir.mkdir(
                parents=True, exist_ok=True
            )
            plt.subplots_adjust(left=0.25) 
            causal_direction = "causal" if causal else "anticausal"
            plt.savefig(
                save_dir / f"pair{pair_index}_{causal_direction}_{save_name}.jpg",
            )
            plt.close()
        else:
            pass

        return conditional_model
