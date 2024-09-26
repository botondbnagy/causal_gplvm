"""
This method will fit a gplvm for both the marginal and conditional models and
choose the causal direction as the one with the minimum
-log marginal likelihood.
"""
from optparse import check_choice
from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.utilities import positive
from models.BayesGPLVM import BayesianGPLVM
from models.PartObsBayesianGPLVM import PartObsBayesianGPLVM
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm import trange
from typing import Optional
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import dill
import tensorflow as tf
import tensorflow_probability as tfp
from collections import defaultdict
from collections import namedtuple


def get_marginal_noise_model_score(
    y: np.ndarray,
    num_inducing: int
):
    # Need to find the ELBO for a noise model
    linear_kernel = gpflow.kernels.Linear(variance=1e-20)
    kernel = linear_kernel

    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, 1),
    )
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], 1)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], 1), dtype=default_float())
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, 1),
    )
    # Define marginal model
    marginal_model = BayesianGPLVM(
        data=y,
        kernel=kernel,
        X_data_mean=X_mean_init,
        X_data_var=X_var_init,
        X_prior_var=x_prior_var,
        jitter=1e-5,
        inducing_variable=inducing_variable
    )
    marginal_model.likelihood.variance = Parameter(
        1.0, transform=positive(1e-6)
    )
    # Train everything
    gpflow.utilities.set_trainable(marginal_model.kernel, True)
    gpflow.utilities.set_trainable(marginal_model.likelihood, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_mean, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_var, True)
    gpflow.utilities.set_trainable(marginal_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        marginal_model.training_loss,
        marginal_model.trainable_variables,
        options=dict(maxiter=10000),
    )
    loss = - marginal_model.elbo()
    tf.print(f"Marginal noise model score: {loss}")
    return loss


def get_conditional_noise_model_score(
    x: np.ndarray,
    y: np.ndarray,
    num_inducing: int
):
    # Need to find the ELBO for a noise model
    linear_kernel = gpflow.kernels.Linear(variance=1)
    kernel = linear_kernel

    Z = gpflow.inducing_variables.InducingPoints(
            np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
        )
    inducing_variable = Z

    reg_gp_model = gpflow.models.SGPR(
        data=(x, y), kernel=kernel, inducing_variable=inducing_variable
    )
    reg_gp_model.likelihood.variance = Parameter(
        1 + 1e-20, transform=positive(lower=1e-6)
    )
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        reg_gp_model.training_loss,
        reg_gp_model.trainable_variables,
        options=dict(maxiter=20000),
    )
    loss = - reg_gp_model.elbo()
    return loss


def train_marginal_model(
    y: np.ndarray,
    num_inducing: int,
    kernel_variance: float,
    kernel_lengthscale: float,
    likelihood_variance: float,
    work_dir: str,
    save_name: str,
    jitter: float,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    opt_iter: Optional[int] = 10000,
    adam_lr: Optional[float] = 0.01,
    plot_fit: Optional[bool] = False,
):
    latent_dim = 1
    # Define kernel
    sq_exp = gpflow.kernels.SquaredExponential(lengthscales=[kernel_lengthscale])
    sq_exp.variance.assign(kernel_variance)
    linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
    # Initialise approx posteroir and prior
    X_mean_init = 0.1 * tf.cast(y, default_float())
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.random.randn(num_inducing, latent_dim),
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

    # We will train the Adam until the elbo gets below the noise model score
    noise_elbo = get_marginal_noise_model_score(
        y=y,
        num_inducing=num_inducing
    )
    loss_fn = marginal_model.training_loss_closure()
    adam_vars = marginal_model.trainable_variables
    adam_opt = tf.optimizers.Adam(adam_lr)
    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)
    epochs = int(opt_iter)

    tf.print("Init: sqe_len: {}, sqe_var: {}, lin_var: {}, like_var: {}".format(
        marginal_model.kernel.kernels[0].lengthscales.numpy(),
        marginal_model.kernel.kernels[0].variance.numpy(),
        marginal_model.kernel.kernels[1].variance.numpy(),
        marginal_model.likelihood.variance.numpy()
    ))
    for epoch in tqdm(range(epochs)):
        optimisation_step()
        if - marginal_model.elbo() < noise_elbo:
            print(f"Breaking as {- marginal_model.elbo()} is less than {noise_elbo}")
            break

    # Train everything
    tf.print("Training everything")
    gpflow.utilities.set_trainable(marginal_model.kernel, True)
    gpflow.utilities.set_trainable(marginal_model.likelihood, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_mean, True)
    gpflow.utilities.set_trainable(marginal_model.X_data_var, True)
    gpflow.utilities.set_trainable(marginal_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    logf = opt.minimize(
        marginal_model.training_loss,
        marginal_model.trainable_variables,
        options=dict(maxiter=opt_iter),
    )

    tf.print("Final: sqe_len: {}, sqe_var: {}, lin_var: {}, like_var: {}".format(
        marginal_model.kernel.kernels[0].lengthscales.numpy(),
        marginal_model.kernel.kernels[0].variance.numpy(),
        marginal_model.kernel.kernels[1].variance.numpy(),
        marginal_model.likelihood.variance.numpy()
    ))
    full_elbo = marginal_model.elbo()

    marg_ll = float(full_elbo)
    pred_loss = float(-marginal_model.predictive_score(y))

    tf.print(f"Full ELBO: {marg_ll}, pred_loss: {pred_loss}")


    if plot_fit:
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
        plt.text(-3.5, 0, textstr, fontsize=8)
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
        fname = f"marginal_" + save_name + ".jpg"
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
        fname = f"marginal_" + save_name + "_hist.jpg"
        plt.savefig(
            save_dir / fname
        )
        plt.close()

        # #plot elbo over iterations with line at 0
        # plt.axhline(y=0, color='black', linestyle='--')
        # logf_arr = np.array(logf)
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
    kernel_lengthscale: float,
    likelihood_variance: float,
    work_dir: str,
    jitter: float,
    save_name: str,
    causal: Optional[bool] = None,
    run_number: Optional[int] = None,
    random_restart_number: Optional[int] = None,
    opt_iter: Optional[int] = 10000,
    adam_lr: Optional[float] = 0.01,
    plot_fit: Optional[bool] = False,
):
    """
    Train a conditional model using a partially observed GPLVM.
    """
    tf.print(f"Init: ker_len: {kernel_lengthscale}, ker_var: {kernel_variance}, like_var: {likelihood_variance}")
    latent_dim = 1

    # Flip coin to see if we should initialise with a GP
    use_gp_initialise = 0
    if use_gp_initialise == 1:
        use_gp = True
    else:
        use_gp = False

    # If use_gp, fit a GP to initialise everything
    if use_gp:
        # Find the best lengthscale for the observed bit
        # Define kernel
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[kernel_lengthscale]
        )
        linear_kernel = gpflow.kernels.Linear(variance=kernel_variance)
        sq_exp.variance.assign(kernel_variance)
        kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
        # Define moedl
        reg_gp_model = gpflow.models.GPR(data=(x, y), kernel=kernel, mean_function=None)
        reg_gp_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(lower=1e-6)
        )
        # Fit model
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(
            reg_gp_model.training_loss, reg_gp_model.trainable_variables, options=dict(maxiter=10000)
        )
        found_lengthscale = float(reg_gp_model.kernel.kernels[0].lengthscales.numpy())
        found_lik_var = reg_gp_model.likelihood.variance.numpy()
        found_kern_var_0 = reg_gp_model.kernel.kernels[0].variance.numpy()
        found_kern_var_1 = reg_gp_model.kernel.kernels[1].variance.numpy()
        tf.print(
            f"Found: ker_len: {found_lengthscale}, ker_var: {found_kern_var_0}, like_var: {found_lik_var}")

        # Put in new values of hyperparams
        X_mean_init = y - reg_gp_model.predict_y(x)[0]
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[found_lengthscale + 1e-20] + [found_lengthscale / 3 + 1e-20]
        )
        sq_exp.variance.assign(found_kern_var_0 + 1e-20)
        linear_kernel = gpflow.kernels.Linear(variance=found_kern_var_1 + 1e-20)
    else:
        # if not using a GP, put in initial values for hyperparams
        X_mean_init = 0.1 * tf.cast(y, default_float())
        sq_exp = gpflow.kernels.SquaredExponential(
            lengthscales=[kernel_lengthscale] + [kernel_lengthscale / 3]
        )
        sq_exp.variance.assign(kernel_variance + 1e-20)
        linear_kernel = gpflow.kernels.Linear(variance=kernel_variance + 1e-20)

    # Define rest of the hyperparams
    X_var_init = tf.cast(
        np.random.uniform(0, 0.1, (y.shape[0], latent_dim)), default_float()
    )
    x_prior_var = tf.ones((y.shape[0], latent_dim), dtype=default_float())
    kernel = gpflow.kernels.Sum([sq_exp, linear_kernel])
    inducing_variable = gpflow.inducing_variables.InducingPoints(
        np.concatenate(
            [
                np.linspace(x.min(), x.max(), num_inducing).reshape(-1, 1),
                np.random.randn(num_inducing, 1),
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
        inducing_variable=inducing_variable
    )
    if use_gp:
        conditional_model.likelihood.variance = Parameter(
            found_lik_var + 1e-20, transform=positive(lower=1e-6)
        )
    else:
        conditional_model.likelihood.variance = Parameter(
            likelihood_variance, transform=positive(lower=1e-6)
        )

    # We will train the Adam until the elbo gets below the noise model score
    noise_elbo = get_conditional_noise_model_score(
        x=x,
        y=y,
        num_inducing=num_inducing
    )
    loss_fn = conditional_model.training_loss_closure()
    adam_vars = conditional_model.trainable_variables
    adam_opt = tf.optimizers.Adam(adam_lr)
    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)
    epochs = int(opt_iter)
    for epoch in tqdm(range(epochs)):
        optimisation_step()
        if - conditional_model.elbo() < noise_elbo:
            print(f"Breaking as {- conditional_model.elbo()} is less than {noise_elbo}")
            break

    # Train everything after Adam
    tf.print("Training everything")
    gpflow.utilities.set_trainable(conditional_model.kernel, True)
    gpflow.utilities.set_trainable(conditional_model.likelihood, True)
    gpflow.utilities.set_trainable(conditional_model.X_data_mean, True)
    gpflow.utilities.set_trainable(conditional_model.X_data_var, True)
    gpflow.utilities.set_trainable(conditional_model.inducing_variable, True)
    opt = gpflow.optimizers.Scipy()
    logf = opt.minimize(
        conditional_model.training_loss,
        conditional_model.trainable_variables,
        options=dict(maxiter=opt_iter),
    )

    full_elbo = conditional_model.elbo()
    marg_ll = float(full_elbo)
    pred_loss = float(-conditional_model.predictive_score((x, y)))
    tf.print(f"Full ELBO: {marg_ll}, pred_loss: {pred_loss}")

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
        
        text_likelihood = 'likelihood: var=%.3f' % (
            conditional_model.likelihood.variance.numpy(),
        )
        textstr = text_sqe + "\n" + text_lin + "\n" + text_likelihood

        #put text on plot
        plt.text(-8.5, 0, textstr, fontsize=8)

        #legend should have alpha 1
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        
        save_dir = Path(f"{work_dir}/figs/run_plots/inf_data")
        save_dir.mkdir(
            parents=True, exist_ok=True
        )
        plt.subplots_adjust(left=0.25) 
        fname = f"conditional_" + save_name + ".jpg"
        plt.savefig(
            save_dir / fname
        )
        plt.close()

        #plot elbo over iterations
        # plt.axhline(y=0, color='black', linestyle='--')
        # logf_arr = np.array(logf)
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


def causal_score_gplvm(
        data,
        num_inducing: int,
        opt_iter: int,
        minibatch_size: int,
        optimiser: str,
        plot_fit: bool,
        dataset_name: str,
        work_dir: Optional[str] = ".",
        causal: Optional[bool] = True,
        random_restart_number: Optional[int] = 0,
        set_size: Optional[int] = None):

    # Unpack data
    x, y = data
    causal_direction = "causal" if causal else "anticausal"
    save_name = f"{causal_direction}_{dataset_name}_size_{set_size}_rr_{random_restart_number}"
    num_inducing = num_inducing if x.shape[0] > num_inducing else x.shape[0]
    run_number = 0
    # Dynamically reduce the jitter if there is an error
    # Sample hyperparams
    kernel_variance = 1
    jitter_bug = 1e-5
    adam_lr = 0.001
    finish = 0
    loss_x = None
    # Likelihood variance
    kappa = np.random.uniform(
        low=10.0, high=100, size=[1]
    )
    likelihood_variance = 1. / (kappa ** 2)
    # Kernel lengthscale
    lamda = np.random.uniform(
        low=1, high=100, size=[1]
    )
    kernel_lengthscale = 1.0 / lamda
    while finish == 0:
        try:
            tf.print("X" if causal else "Y")
            marg_ll_x, pred_loss_x = train_marginal_model(
                y=x,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
                work_dir=work_dir,
                run_number=run_number,
                random_restart_number=random_restart_number,
                jitter=jitter_bug,
                causal=causal,
                save_name=save_name,
                opt_iter=opt_iter,
                adam_lr=adam_lr,
                plot_fit=plot_fit,
            )
            finish = 1
        except Exception as e:
            tf.print(e)
            tf.print(f"Increasing jitter to {jitter_bug * 10}")
            jitter_bug *= 10
            if jitter_bug > 1:
                finish = 1
                raise ValueError("jitter is more than 1!")
            
    # Sample hyperparams
    jitter_bug = 1e-5
    finish = 0
    kernel_variance = 1
    # Likelihood variance
    kappa = np.random.uniform(
        low=10.0, high=100, size=[1]
    )
    likelihood_variance = 1. / (kappa ** 2)
    # Kernel lengthscale
    lamda = np.random.uniform(
        low=1.0, high=100, size=[1]
    )
    kernel_lengthscale = 1.0 / lamda
    while finish == 0:
        try:
            tf.print("Y|X" if causal else "X|Y")
            marg_ll_y_x, pred_loss_y_x = train_conditional_model(
                x=x,
                y=y,
                num_inducing=num_inducing,
                kernel_variance=kernel_variance,
                kernel_lengthscale=kernel_lengthscale[0],
                likelihood_variance=likelihood_variance[0],
                work_dir=work_dir,
                run_number=run_number,
                random_restart_number=random_restart_number,
                causal=causal,
                jitter=jitter_bug,
                save_name=save_name,
                opt_iter=opt_iter,
                adam_lr=adam_lr,
                plot_fit=plot_fit,
            )
            finish = 1
        except Exception as e:
            tf.print(e)
            jitter_bug *= 10
            tf.print(f"Increasing jitter to {jitter_bug}")
            
            if jitter_bug > 1:
                finish = 1
                raise ValueError("jitter is more than 1!")
            
    return (marg_ll_x, marg_ll_y_x), (pred_loss_x, pred_loss_y_x)