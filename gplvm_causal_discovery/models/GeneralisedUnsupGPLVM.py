"""
This is the unsupervised version of the Generalised GPLVM.
"""
from typing import Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP
from gpflow import likelihoods
from gpflow.base import Parameter
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.config import default_float
from gpflow.utilities import positive, to_default_float
from gpflow.models.gpr import GPR
from gpflow.models.model import MeanAndVariance
from gpflow.models.util import inducingpoint_wrapper
from gpflow.conditionals.util import sample_mvn


class GeneralisedUnsupGPLVM(SVGP):
    def __init__(
        self,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        likelihood: likelihoods,
        num_mc_samples: int,
        inducing_variable: InducingVariables,
        batch_size: int,
        q_mu=None,
        q_sqrt=None,
        X_prior_mean=None,
        X_prior_var=None,
    ):
        """
        This is the Generalised GPLVM as listed in:
        https://arxiv.org/pdf/2202.12979.pdf

        The key point here are:
        - Uses uncollapsed inducing variables which allows for minibatching
        - Still computes the kernel expectation, but using MC expectation

        :param X_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x (L + Q) (latent dimensions). By default
            random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        num_data, num_latent_gps = X_data_mean.shape
        
        # push X through encoder (tensorflow NN) constraining 
        # the latent points to be a function of the input points

        # encoder = tf.keras.Sequential([
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(32, activation='relu'),
        #     tf.keras.layers.Dense(num_latent_gps, activation='sigmoid')
        # ])
        # X_data_mean = encoder(X_data_mean)



        
        super().__init__(
            kernel,
            likelihood,
            mean_function=None,
            num_latent_gps=num_latent_gps,
            q_diag=False,
            whiten=True,
            inducing_variable=inducing_variable,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
        )

        # Set in data to be a non trainable parameter
        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.batch_size = batch_size

        self.num_data = num_data
        self.num_mc_samples = num_mc_samples

        assert np.all(X_data_mean.shape == X_data_var.shape)

        if inducing_variable is None:
            raise ValueError("BayesianGPLVM needs `inducing_variable`")

        # Make only the non latent part of inducing trainable
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros(
                (self.num_data, self.num_latent_gps), dtype=default_float()
            )
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(
            np.atleast_1d(X_prior_mean), dtype=default_float()
        )
        self.X_prior_var = tf.convert_to_tensor(
            np.atleast_1d(X_prior_var), dtype=default_float()
        )

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent_gps

    def get_new_mean_vars(self, data_idx) -> MeanAndVariance:
        batch_X_means = tf.gather(self.X_data_mean, data_idx)
        batch_X_vars = tf.gather(self.X_data_var, data_idx)
        new_mean = batch_X_means
        new_variance = batch_X_vars
        return new_mean, new_variance, batch_X_means, batch_X_vars

    def predict_f_samples(
        self,
        Xnew,
        obs_noise=False,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        :param Xnew: InputData
            Input locations at which to draw samples, shape [..., N, D]
            where N is the number of rows and D is the input dimension of each point.
        :param num_samples:
            Number of samples to draw.
            If `None`, a single sample is drawn and the return shape is [..., N, P],
            for any positive integer the return shape contains an extra batch
            dimension, [..., S, N, P], with S = num_samples and P is the number of outputs.
        :param full_cov:
            If True, draw correlated samples over the inputs. Computes the Cholesky over the
            dense covariance matrix of size [num_data, num_data].
            If False, draw samples that are uncorrelated over the inputs.
        :param full_output_cov:
            If True, draw correlated samples over the outputs.
            If False, draw samples that are uncorrelated over the outputs.
        Currently, the method does not support `full_output_cov=True` and `full_cov=True`.
        """
        if full_cov and full_output_cov:
            raise NotImplementedError(
                "The combination of both `full_cov` and `full_output_cov` is not supported."
            )

        # check below for shape info
        mean, cov = self.predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        if obs_noise is True and full_cov is False:
            cov += self.likelihood.variance
        if full_cov:
            # mean: [..., N, P]
            # cov: [..., P, N, N]
            mean_for_sample = tf.linalg.adjoint(mean)  # [..., P, N]
            samples = sample_mvn(
                mean_for_sample, cov, full_cov, num_samples=num_samples
            )  # [..., (S), P, N]
            samples = tf.linalg.adjoint(samples)  # [..., (S), N, P]
        else:
            # mean: [..., N, P]
            # cov: [..., N, P] or [..., N, P, P]
            samples = sample_mvn(
                mean, cov, full_output_cov, num_samples=num_samples
            )  # [..., (S), N, P]
        return samples  # [..., (S), N, P]

    def predict_full_samples_layer(
        self,
        sample_size=1000,
        obs_noise=False,
        num_latent_samples=50,
        num_gp_samples=50,
    ):
        w = np.random.normal(size=(num_latent_samples, sample_size, 1))
        sampling_func = lambda x: self.predict_f_samples(
            x, obs_noise=obs_noise, num_samples=num_gp_samples
        )

        def sample_latent_gp(w_single):
            X = w_single
            samples = sampling_func(X)
            return samples

        samples = tf.map_fn(sample_latent_gp, w)

        return samples

    def predict_credible_layer(
        self,
        sample_size,
        lower_quantile=2.5,
        upper_quantile=97.5,
        num_gp_samples=50,
        num_latent_samples=50,
        obs_noise=False,
    ):

        samples = self.predict_full_samples_layer(
            sample_size=sample_size,
            obs_noise=obs_noise,
            num_gp_samples=num_gp_samples,
            num_latent_samples=num_latent_samples,
        )
        lower = np.percentile(samples, lower_quantile, axis=[0, 1])
        median = np.percentile(samples, 50, axis=[0, 1])
        upper = np.percentile(samples, upper_quantile, axis=[0, 1])

        return lower, median, upper, samples

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data, data_idx = data
        (
            new_mean,
            new_variance,
            batch_X_means,
            batch_X_vars,
        ) = self.get_new_mean_vars(data_idx)

        # We integrate out the latent variable by taking J MC samples
        # [num_mc, num_batch, num_dim]
        X_samples = tfp.distributions.MultivariateNormalDiag(
            loc=new_mean,
            scale_diag=new_variance**0.5,
        ).sample(self.num_mc_samples)
        X_samples = tf.reshape(X_samples, [-1, new_mean.shape[1]])

        # # KL[q(x) || p(x)]
        batch_prior_means = tf.gather(self.X_prior_mean, data_idx)
        batch_prior_vars = tf.gather(self.X_prior_var, data_idx)
        dX_data_var = (
            batch_X_vars
            if batch_X_vars.shape.ndims == 2
            else tf.linalg.diag_part(batch_X_vars)
        )
        NQ = to_default_float(tf.size(batch_X_means))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var + 1e-30))
        KL += 0.5 * tf.reduce_sum(tf.math.log(batch_prior_vars + 1e-30))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(batch_X_means - batch_prior_means) + dX_data_var)
            / batch_prior_vars
        )

        KL_2 = self.prior_kl()
        f_mean, f_var = self.predict_f(
            X_samples, full_cov=False, full_output_cov=False
        )
        f_mean = tf.reshape(
            f_mean, [self.num_mc_samples, -1, self.num_latent_gps]
        )
        f_var = tf.reshape(
            f_var, [self.num_mc_samples, -1, self.num_latent_gps]
        )
        var_exp = self.likelihood.variational_expectations(
            f_mean, f_var, Y_data
        )
        # MC over 1st dim
        var_exp = tf.reduce_mean(var_exp, axis=0)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, KL_2.dtype)
            minibatch_size = tf.cast(tf.shape(Y_data)[0], KL_2.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, KL_2.dtype)
        return (tf.reduce_sum(var_exp) - KL) * scale - KL_2

    def predictive_score(self, data):
        Y_data, data_idx = data
        (
            new_mean,
            new_variance,
            batch_X_means,
            batch_X_vars,
        ) = self.get_new_mean_vars(data_idx)

        samples = self.predict_full_samples_layer(
            sample_size=Y_data.shape[0], obs_noise=True
        )
        normal_dist = tfp.distributions.Normal(
            loc=samples, scale=self.likelihood.variance**0.5
        )
        prob_samples = normal_dist.prob(Y_data)
        mc_samples = tf.math.log(
            tf.reduce_mean(prob_samples, axis=[0, 1])# + 1e-20
                           )
        return tf.reduce_sum(mc_samples)
