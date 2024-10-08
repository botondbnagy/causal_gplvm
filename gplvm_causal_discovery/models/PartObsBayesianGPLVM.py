from tabnanny import check
from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow import covariances, kernels, likelihoods
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.expectations import expectation
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.probability_distributions import DiagonalGaussian
from gpflow.utilities import positive, to_default_float
from gpflow.utilities.ops import pca_reduce
from gpflow.models.gpr import GPR
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, OutputData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

import tensorflow_probability as tfp
# from ops import cholesky
from gpflow.conditionals.util import sample_mvn


class PartObsBayesianGPLVM(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: OutputData,
        in_data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        jitter: float,
        num_inducing_variables: Optional[int] = None,
        inducing_variable=None,
        X_prior_mean=None,
        X_prior_var=None,
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param in_data: input data matrix, size N (number of points) x L (observed dimnensions)
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
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)
        assert X_data_var.ndim == 2

        # Set in data to be a non trainable parameter
        self.in_data = Parameter(in_data, trainable=False)
        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.num_data = num_data
        self.output_dim = self.data.shape[-1]
        self.jitter = jitter

        assert np.all(X_data_mean.shape == X_data_var.shape)
        assert X_data_mean.shape[0] == self.data.shape[0], "X mean and Y must be same size."
        assert X_data_var.shape[0] == self.data.shape[0], "X var and Y must be same size."

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
            # Inducing variables will be a concatenation of X_obs and X_latent
            X_full = tf.concat([self.in_data, X_data_mean], axis=1)
            Z = tf.random.shuffle(X_full)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        # Make only the non latent part of inducing trainable
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent_gps
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent_gps

    def get_new_mean_vars(self) -> MeanAndVariance:
        new_mean = tf.concat(
            [self.in_data, self.X_data_mean], axis=1
        )
        # Set observed variance to 0
        var_param = tf.zeros(shape=self.in_data.shape, dtype=default_float())
        new_variance = tf.concat(
            [var_param, self.X_data_var], axis=1
        )
        return new_mean, new_variance

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        # print(self.kernel.lengthscales)
        Y_data = self.data
        new_mean, new_variance = self.get_new_mean_vars()
        pX = DiagonalGaussian(new_mean, new_variance)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        # try:
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        # except:
        # tf.print(self.kernel.lengthscales, self.kernel.variance, self.likelihood.variance)
        # tf.print(tf.reduce_min(new_variance[:, 1]))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter)
        L = tf.linalg.cholesky(cov_uu)
        # L = cholesky(cov_uu)
        tf.debugging.assert_all_finite(
            L, message="L is not finite!"
        )
        sigma2 = self.likelihood.variance
        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        # LB = cholesky(B)
        tf.debugging.assert_all_finite(
            LB, message="LB is not finite!"
        )
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2

        # KL[q(x) || p(x)]
        dX_data_var = (
            self.X_data_var
            if self.X_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.X_data_var)
        )
        NQ = to_default_float(tf.size(self.X_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var + 1e-30))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var + 1e-30))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        # if np.isnan(LB.numpy()).sum() > 0:
        #     print(f"{self.kernel.variance.numpy()}, {self.likelihood.variance.numpy()}, {self.kernel.lengthscales.numpy()}")
            # import pdb; pdb.set_trace()

        # print(f"{self.kernel.variance.numpy()}, {self.likelihood.variance.numpy()}, {self.kernel.lengthscales.numpy()}")
        return bound

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.

        Note: This model does not allow full output covariances.

        :param Xnew: points at which to predict
        """
        if full_output_cov:
            raise NotImplementedError

        new_mean, new_variance = self.get_new_mean_vars()
        pX = DiagonalGaussian(new_mean, new_variance)

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=self.jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), axis=0)
                - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    def predict_log_density(self, data: OutputData) -> tf.Tensor:
        raise NotImplementedError

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
        mean, cov = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
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

    def predict_full_samples_layer(self, Xnew, obs_noise=False, num_latent_samples=50, num_gp_samples=50):
        w = np.random.normal(size=(num_latent_samples, Xnew.shape[0], 1))
        sampling_func = lambda x: self.predict_f_samples(x, obs_noise=obs_noise, num_samples=num_gp_samples)

        def sample_latent_gp(w_single):
            X = np.concatenate([Xnew, w_single], axis=1)
            samples = sampling_func(X)
            return samples

        samples = tf.map_fn(sample_latent_gp, w)

        return samples

    def predict_credible_layer(
        self,
        Xnew,
        lower_quantile=2.5,
        upper_quantile=97.5,
        num_gp_samples=50,
        num_latent_samples=50,
        obs_noise=False,
    ):

        samples = self.predict_full_samples_layer(
            Xnew,
            obs_noise=obs_noise,
            num_gp_samples=num_gp_samples,
            num_latent_samples=num_latent_samples,
        )
        lower = np.percentile(samples, lower_quantile, axis=[0,1])
        median = np.percentile(samples, 50, axis=[0,1])
        upper = np.percentile(samples, upper_quantile, axis=[0,1])

        return lower, median, upper, samples

    def predictive_score(self, data):
        """
        Just return the predictive part of the loss.
        """
        X_data, Y_data = data
        samples = self.predict_full_samples_layer(
            X_data, obs_noise=False
        )
        normal_dist = tfp.distributions.Normal(
            loc=samples,
            scale=self.likelihood.variance ** 0.5
        )
        prob_samples = normal_dist.prob(Y_data)
        mc_samples = tf.math.log(
            tf.reduce_mean(prob_samples, axis=[0, 1]) + 1e-20
        )
        return tf.reduce_sum(mc_samples)
