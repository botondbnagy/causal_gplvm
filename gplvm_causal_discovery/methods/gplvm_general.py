"""
This method fits a GPLVM for marginal and conditional models.
Returns the marginal log likelihood and predictive loss for each
"""
from models.GeneralisedGPLVM import GeneralisedGPLVM
from models.GeneralisedUnsupGPLVM import GeneralisedUnsupGPLVM
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Optional
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
from gpflow.config import default_float
import dill
from collections import defaultdict
from collections import namedtuple
import wandb

def run_optimizer(
    optimiser, model, train_dataset, iterations, data_size, minibatch_size, adam_lr
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
    # start a new wandb run to track this script
    wandb.init(
        project="causal_cross_validation",

        # track hyperparameters and run metadata
        config={
            "optimiser": optimiser,
            "iterations": iterations,
            "data_size": data_size,
            "minibatch_size": minibatch_size,
            "learning_rate": adam_lr,
        }
    )

    if optimiser == "adam":
        optimizer = tf.optimizers.Adam(adam_lr)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)

    elif optimiser == "natgrad":
        natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=adam_lr)
        variational_params = [(model.q_mu, model.q_sqrt)]

        #prohibit training the variational parameters for Adam
        gpflow.utilities.set_trainable(model.q_mu, False)
        gpflow.utilities.set_trainable(model.q_sqrt, False)

        adam_opt = tf.optimizers.Adam(adam_lr)
        
        @tf.function
        def optimization_step():
            natgrad_opt.minimize(
                training_loss, var_list=variational_params
            )
            adam_opt.minimize(training_loss, var_list=model.trainable_variables)

    else:
        raise ValueError("optimiser must be 'adam' or 'natgrad'")

    iterator = range(iterations)
    for step in tqdm(iterator, desc=f"Optimising with {optimiser}"):
        optimization_step()
        neg_elbo = training_loss().numpy()
        logf.append(neg_elbo)
        wandb.log({"neg_elbo": neg_elbo})
        # It is possible to include a stopping criteria here. However there
        # is a risk that the training will be stopped too soon and the
        # ELBO achieved will not be as high as it could be

    wandb.finish()
    return logf


def train_marginal_model(
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale_1: float,
    likelihood_variance: float,
    num_minibatch: int,
    num_iterations: int, 
    optimiser: str,
    work_dir: str,
    save_name: str,
    adam_lr: float,
    causal: Optional[bool] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    
    # Set random seed
    causal_offset = 1 if causal else 10
    size_offset = y.shape[0]
    seed = causal_offset * (size_offset + random_restart_number)
    rng = np.random.default_rng(seed)

    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=kernel_lengthscale_1
    )
    sq_exp.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
    # Z = np.random.randn(num_inducing, 1)
    Z = rng.normal(size=(num_inducing, 1))
    # inducing_variable = gpflow.inducing_variables.InducingPoints(
    #     # np.random.randn(num_inducing, 1)
    #     Z
    # )
    #Z = inducing_variable
    # Define the approx posteroir
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        # np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
        rng.uniform(0, 0.1, (y.shape[0], 1)), default_float()
    )

    # Define marginal model
    marginal_model = GeneralisedUnsupGPLVM(
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=likelihood_variance),
        num_mc_samples=10,
        inducing_variable=Z,
        batch_size=num_minibatch,
    )
    # Run optimisation
    tf.print("Init: sqe_len={}, sqe_var={}, lin_var={}, lik_var={}".format(
        marginal_model.kernel.kernels[0].lengthscales.numpy(),
        marginal_model.kernel.kernels[0].variance.numpy(),
        marginal_model.kernel.kernels[1].variance.numpy(),
        marginal_model.likelihood.variance.numpy(),
    ))

    data_idx = np.arange(y.shape[0])
    train_dataset = tf.data.Dataset.from_tensor_slices((y, data_idx)).repeat()
    logf = run_optimizer(
        optimiser=optimiser,
        model=marginal_model,
        train_dataset=train_dataset,
        iterations=num_iterations,
        adam_lr=adam_lr,
        data_size=y.shape[0],
        minibatch_size=num_minibatch,
    )
    tf.print("Found: sqe_len={}, sqe_var={}, lin_var={}, lik_var={}".format(
        marginal_model.kernel.kernels[0].lengthscales.numpy(),
        marginal_model.kernel.kernels[0].variance.numpy(),
        marginal_model.kernel.kernels[1].variance.numpy(),
        marginal_model.likelihood.variance.numpy(),
    ))

    marginal_model.num_mc_samples = 100
    full_elbo = marginal_model.elbo((y, data_idx))
    tf.print(f"Full Loss: {- full_elbo}")

    marg_ll = float(full_elbo)
    pred_loss = float(-marginal_model.predictive_score((y, data_idx)))


    if plot_fit:
        # Plot the fit to see if everything is ok
        num_lat = 20
        num_gp = 50
        
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
        text_likelihood = 'likelihood_var=%.3f' % (
            marginal_model.likelihood.variance.numpy(),
        )
        textstr = text_sqe + '\n' + text_lin + '\n' + text_likelihood

        plt.text(obs_new.min(), pred_y_mean.min()+1, textstr, fontsize=8)
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
        fname = f"marginal_" + save_name + ".pdf"
        save_dir = Path(f"{work_dir}/figs/run_plots/inf_data")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25)
        
        plt.savefig(
            save_dir / fname
        )
        plt.close()

        #histogram of y observed
        plt.hist(y, bins=50, density=True, color="C2", alpha=0.5, label="data")

        #histogram of samples from posterior
        samples = marginal_model.predict_full_samples_layer(sample_size=1000,
                                                            obs_noise=True, 
                                                            num_latent_samples=num_lat, 
                                                            num_gp_samples=num_gp)
        samples = samples.numpy().reshape(num_gp, num_lat, -1)
        samples = samples.flatten()
        plt.hist(samples, bins=50, density=True, color="C0", alpha=0.5, label="samples")
        plt.legend()
        fname = f"marginal_" + save_name + "_hist.pdf"
        plt.savefig(
            save_dir / fname
        )
        plt.close()

        #plot elbo over iterations with line at 0
        # plt.axhline(y=0, color='black', linestyle='--')
        # logf_arr = np.array(logf)[-1000:]
        # iters_arr = np.arange(len(logf)-len(logf_arr), len(logf))
        # plt.plot(iters_arr, logf_arr)
        # plt.xlabel("iteration")
        # plt.ylabel("negated ELBO")
        # fname = f"marginal_" + save_name + "_elbo.jpg"
        # plt.savefig(
        #     save_dir / fname
        # )
        # plt.close()

    else:
        pass
    return marg_ll, pred_loss


def train_conditional_model(
    x: np.ndarray,
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale_1: float,
    likelihood_variance: float,
    num_minibatch: int,
    num_iterations: int,
    optimiser: str,
    work_dir: str,
    save_name: str,
    adam_lr: float,
    causal: Optional[bool] = None,
    random_restart_number: Optional[int] = None,
    plot_fit: Optional[bool] = False,
):
    """
    Train a conditional model using a partially observed GPLVM.
    """

    # Set random seed
    causal_offset = 1 if causal else 10
    size_offset = x.shape[0]
    seed = causal_offset * (size_offset + random_restart_number)
    rng = np.random.default_rng(seed)
    # Define kernels
    sq_exp = gpflow.kernels.SquaredExponential(
        lengthscales=[kernel_lengthscale_1, kernel_lengthscale_1 * 0.3]
    )
    sq_exp.variance.assign(kernel_variance)

    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])

    Z = np.concatenate(
        [
            np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
            # np.random.randn(num_inducing, 1),
            rng.normal(size=(num_inducing, 1)),
        ],
        axis=1,
    )

    # Define the approx posteroir
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        # np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
        rng.uniform(0, 0.1, (y.shape[0], 1)), default_float()
    )

    # Define the conditional model
    conditional_model = GeneralisedGPLVM(
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=likelihood_variance),
        num_mc_samples=10,
        inducing_variable=Z,
        batch_size=num_minibatch,
    )

    # Run optimisation
    tf.print("Init: sqe_len={}, sqe_var={}, lin_var={}, lik_var={}".format(
        conditional_model.kernel.kernels[0].lengthscales.numpy(),
        conditional_model.kernel.kernels[0].variance.numpy(),
        conditional_model.kernel.kernels[1].variance.numpy(),
        conditional_model.likelihood.variance.numpy(),
    ))
    data_idx = np.arange(y.shape[0])
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x, y, data_idx)
    ).repeat()
    logf = run_optimizer(
        optimiser=optimiser,
        model=conditional_model,
        train_dataset=train_dataset,
        iterations=num_iterations,
        adam_lr=adam_lr,
        data_size=y.shape[0],
        minibatch_size=num_minibatch,
    )
    tf.print("Found: sqe_len={}, sqe_var={}, lin_var={}, lik_var={}".format(
        conditional_model.kernel.kernels[0].lengthscales.numpy(),
        conditional_model.kernel.kernels[0].variance.numpy(),
        conditional_model.kernel.kernels[1].variance.numpy(),
        conditional_model.likelihood.variance.numpy(),
    ))

    conditional_model.num_mc_samples = 100
    full_elbo = conditional_model.elbo((x, y, data_idx))
    tf.print(f"Full Loss: {- full_elbo}")

    marg_ll = float(full_elbo)
    pred_loss = float(-conditional_model.predictive_score((x, y, data_idx)))

    if plot_fit:
        # Plot the fit to see if everything is ok
        num_lat = 1
        num_gp = 1
        Xnew = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
        samples = conditional_model.predict_full_samples_layer(Xnew,
                                                        obs_noise=True,
                                                        num_latent_samples=num_lat,
                                                        num_gp_samples=num_gp)
        
        samples = samples.numpy().reshape(num_gp, num_lat, -1)
        
        Xnew = np.tile(Xnew, (num_gp, num_lat, 1))
        
        plt.scatter(Xnew, samples, color="C0", alpha=0.5, label="samples", marker=".", edgecolors="none", s=100)
        plt.scatter(x, y, color="C2", label="data", marker=".", alpha=0.5, edgecolors="none", s=100)

        text_sqe = 'sqe: len_obs=%.3f, len_lat=%.3f, var=%.3f' % (
            conditional_model.kernel.kernels[0].lengthscales.numpy()[0],
            conditional_model.kernel.kernels[0].lengthscales.numpy()[1],
            conditional_model.kernel.kernels[0].variance.numpy(),
        )
        text_lin = 'lin: var=%.3f' % (
            conditional_model.kernel.kernels[1].variance.numpy(),
        )
        
        text_likelihood = 'likelihood: var=%.3f' % (
            conditional_model.likelihood.variance.numpy(),
        )
        textstr = text_sqe + "\n" + text_lin + "\n" + text_likelihood

        #put text on plot
        plt.text(x.min(), y.min()+1, textstr, fontsize=8)

        #legend should have alpha 1
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        
        save_dir = Path(f"{work_dir}/figs/run_plots/inf_data")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25) 
        fname = f"conditional_" + save_name + ".pdf"
        plt.savefig(
            save_dir / fname
        )
        plt.close()

        #plot elbo over iterations
        # plt.axhline(y=0, color='black', linestyle='--')
        # logf_arr = np.array(logf)[-1000:]
        # iters_arr = np.arange(len(logf)-len(logf_arr), len(logf))
        # plt.plot(iters_arr, logf_arr)
        # plt.xlabel("iteration")
        # plt.ylabel("negated ELBO")
        # fname = f"conditional_" + save_name + "_elbo.jpg"
        # plt.savefig(
        #     save_dir / fname
        # )
        # plt.close()
    else:
        pass

    return marg_ll, pred_loss


def causal_scores(
    data,
    num_inducing,
    opt_iter,
    minibatch_size,
    set_size,
    plot_fit,
    optimiser,
    dataset_name: str="",
    work_dir: Optional[str] = ".",
    causal: Optional[bool] = True,
    random_restart_number: Optional[int] = 0,
):
    # Unpack data
    x, y = data
    causal_direction = "causal" if causal else "anticausal"
    save_name = f"{causal_direction}_{dataset_name}_size_{set_size}_ind_{num_inducing}_opt_{optimiser}_rr_{random_restart_number}"
    num_inducing = (
        num_inducing if x.shape[0] > num_inducing else x.shape[0]
    )

    # Set random seed
    causal_offset = 1 if causal else 10
    size_offset = set_size
    seed = causal_offset * (size_offset + random_restart_number)
    rng = np.random.default_rng(seed)
    # Set number of iterations
    num_iterations = opt_iter

    # Sample hyperparams
    kernel_variance = rng.uniform(low=0.1, high=1.0)
    # Likelihood variance
    kappa = rng.uniform(low=50.0, high=100.0, size=[1])
    likelihood_variance = 1.0 / (kappa**2)
    # Kernel lengthscale
    lamda = rng.uniform(low=1.0, high=2.0, size=[3])
    kernel_lengthscale = 1.0 / lamda
    adam_lr = 0.1

    tf.print("X" if causal else "Y")
    marg_ll_x, pred_loss_x = train_marginal_model(
        y=x,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale_1=kernel_lengthscale[0],
        likelihood_variance=likelihood_variance[0],
        num_minibatch=minibatch_size,
        optimiser=optimiser,
        work_dir=work_dir,
        random_restart_number=random_restart_number,
        causal=causal,
        save_name=save_name,
        plot_fit=plot_fit,
        adam_lr=adam_lr,
        num_iterations=num_iterations,
    )

    # Sample hyperparams
    kernel_variance = rng.uniform(low=0.1, high=1.0)
    # Likelihood variance
    kappa = rng.uniform(low=2.0, high=10.0, size=[1])
    likelihood_variance = 1.0 / (kappa**2)
    # Kernel lengthscale
    lamda = rng.uniform(low=1.0, high=2.0, size=[3])
    kernel_lengthscale = 1.0 / lamda
    adam_lr = 0.1

    tf.print("Y|X" if causal else "X|Y")
    marg_ll_y_x, pred_loss_y_x = train_conditional_model(
        x=x,
        y=y,
        num_inducing=num_inducing,
        kernel_variance=kernel_variance,
        kernel_lengthscale_1=kernel_lengthscale[0],
        likelihood_variance=likelihood_variance[0],
        num_minibatch=minibatch_size,
        optimiser=optimiser,
        work_dir=work_dir,
        random_restart_number=random_restart_number,
        causal=causal,
        save_name=save_name,
        plot_fit=plot_fit,
        adam_lr=adam_lr,
        num_iterations=num_iterations,
    )

    return (marg_ll_x, marg_ll_y_x), (pred_loss_x, pred_loss_y_x)