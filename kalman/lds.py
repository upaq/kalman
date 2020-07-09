import numpy as np
import pandas as pd
import pybasicbayes
import pylds.models

from pybasicbayes.util.stats import sample_mniw
from typing import Tuple

from .dynamic_bias import AbstractDynamicBias


def VanillaLDS(D_obs, D_latent, D_input=0,
               mu_init=None, sigma_init=None,
               A=None, B=None, sigma_states=None,
               C=None, D=None, sigma_obs=None,
               nu_0_dyn=None, S_0_dyn=None, M_0_dyn=None, K_0_dyn=None,
               nu_0_em=None, S_0_em=None, M_0_em=None, K_0_em=None):

    # y | A, x, Sigma ~ N(y ; Ax, Sigma)
    # See page 72 in Gelman et al's Bayesian Data Analysis (3rd ed):
    #
    # Sigma    ~ IW(Sigma ; S_0, nu_0)
    # nu_0     : degrees of freedom
    # S_0      : scale matrix
    # E[Sigma] = 1 / (nu_0 - dim - 1) * S_0
    # With nu_0 = dim + 1 degrees of freedom, each of the correlations in Sigma
    # has a marginal uniform prior distribution (although the joint, with Sigma
    # being positive definite, doesn't).
    #
    # A | Sigma ~ N(A ; M_0, Sigma K_0)
    # M_0       : mean of A
    # K_0       : covariance scaling of rows of A

    if nu_0_dyn is None:
        nu_0_dyn = D_latent + 1
    if S_0_dyn is None:
        S_0_dyn = D_latent * np.eye(D_latent)
    if M_0_dyn is None:
        M_0_dyn = np.zeros((D_latent, D_latent + D_input))
    if K_0_dyn is None:
        K_0_dyn = D_latent * np.eye(D_latent + D_input)
    if nu_0_em is None:
        nu_0_em = D_obs + 1
    if S_0_em is None:
        S_0_em = D_obs * np.eye(D_obs)
    if M_0_em is None:
        M_0_em = np.zeros((D_obs, D_latent + D_input))
    if K_0_em is None:
        K_0_em = D_obs * np.eye(D_latent + D_input)

    model = pylds.models.LDS(
        dynamics_distn=pybasicbayes.distributions.Regression(
            nu_0=nu_0_dyn, S_0=S_0_dyn, M_0=M_0_dyn, K_0=K_0_dyn),
        emission_distn=pybasicbayes.distributions.Regression(
            nu_0=nu_0_em, S_0=S_0_em, M_0=M_0_em, K_0=K_0_em))

    set_default = \
        lambda prm, val, default: \
            model.__setattr__(prm, val if val is not None else default)

    set_default('mu_init', mu_init, np.zeros(D_latent))
    set_default('sigma_init', sigma_init, np.eye(D_latent))

    set_default('A', A, 0.99 * pylds.util.random_rotation(D_latent))
    set_default('B', B, 0.1 * np.random.randn(D_latent, D_input))
    set_default('sigma_states', sigma_states, 0.1 * np.eye(D_latent))

    set_default('C', C, np.random.randn(D_obs, D_latent))
    set_default('D', D, 0.1 * np.random.randn(D_obs, D_input))
    set_default('sigma_obs', sigma_obs, 0.1 * np.eye(D_obs))

    return model


class LDS(object):
    def __init__(self, D_obs: int, D_latent: int, D_input: int,
                 D_input_gamma: int=0, dynamics_prior_scalar: float=1.0):
        assert D_input_gamma <= D_input

        self.D_obs = D_obs
        self.D_latent = D_latent
        self.D_input = D_input
        self.D_input_gamma = D_input_gamma

        self.C_constant = np.zeros((self.D_obs, self.D_latent))
        self.C_constant[:, -1] = 1

        # We let gamma ~ N(1, 1) distribution. It scales a subset of the
        # external controls (e.g. the Oxford government stringency index)
        # per country or time series. A mean at one (and not zero) breaks
        # symmetry; we could alternatively place a prior on it to enforce
        # non-negativity.
        self.gamma_prior_precision = 1
        self.gamma_prior_mean_times_precision = 1

        mu_init = np.zeros(D_latent)
        sigma_init = np.eye(D_latent)

        K_0_dyn = dynamics_prior_scalar * D_latent * np.eye(D_latent + D_input)
        self.model = VanillaLDS(self.D_obs, self.D_latent, self.D_input,
                                A=np.eye(self.D_latent),
                                B=np.zeros((self.D_latent, self.D_input)),
                                C=np.ones((self.D_obs, self.D_latent)),
                                D=np.zeros((self.D_obs, self.D_input)),
                                mu_init=mu_init,
                                sigma_init=sigma_init,
                                K_0_dyn=K_0_dyn)
        self.gammas = []
        self.inputs = []

    def set_parameters(self, mu_init=None, sigma_init=None,
                       A=None, B=None, sigma_states=None,
                       C=None, D=None, sigma_obs=None,
                       gamma_prior_precision=None,
                       gamma_prior_mean_times_precision=None):
        if mu_init is not None:
            self.model.mu_init = mu_init
        if sigma_init is not None:
            self.model.sigma_init = sigma_init
        if A is not None:
            self.model.A = A
        if B is not None:
            self.model.B = B
        if C is not None:
            self.model.C = C
        if D is not None:
            self.model.D = D
        if sigma_states is not None:
            self.model.dynamics_distn.sigma = sigma_states
        if sigma_obs is not None:
            self.model.emission_distn.sigma = sigma_obs
        if gamma_prior_precision is not None:
            self.gamma_prior_precision = gamma_prior_precision
        if gamma_prior_mean_times_precision is not None:
            self.gamma_prior_mean_times_precision = \
                gamma_prior_mean_times_precision

    def add_data(self, x, inputs, gamma=1.0):
        self.model.add_data(x, inputs=inputs)
        self.gammas.append(gamma)
        self.inputs.append(inputs)

    def pop_time_series(self) -> Tuple[pylds.states.LDSStates,
                                       float,
                                       np.ndarray]:
        """Removes the last time series, added with `add_data(...)`, from the
        model."""
        s = self.model.states_list.pop()
        gamma = self.gammas.pop()
        inputs = self.inputs.pop()
        return s, gamma, inputs

    def infer_and_forward_sample(self,
                                 x: np.ndarray,
                                 inputs: np.ndarray,
                                 gamma: float,
                                 Tpred: int,
                                 future_inputs: np.ndarray=None,
                                 dynamic_bias: AbstractDynamicBias=None,
                                 country_df: pd.DataFrame=None,
                                 states_noise: bool=False,
                                 obs_noise: bool=False) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
        """Infers the posterior marginal means and covariances of p(z|x),
        given the current model parameters. Returns the posterior predictive
        means and covariances of x (for error bars). Forward-samples `Tpred`
        future observations, starting with the filtered posterior mean and
        covariance of the last element of p(z|x) as prior.

        Code adapted from pylds/pylds/states.py"""

        self.add_data(x, inputs, gamma)

        if self.D_input_gamma > 0:
            self._scale_inputs_by_gamma()

        self.model.resample_states()

        s, _, _ = self.pop_time_series()

        s.smooth()

        smoothed_x_mean = s.smoothed_mus.dot(s.C.T) + s.inputs.dot(s.D.T)

        C = np.expand_dims(s.C, axis=0)
        smoothed_x_cov = np.matmul(C, np.matmul(s.smoothed_sigmas,
                                                np.transpose(C, (0, 2, 1))))
        smoothed_x_cov += np.expand_dims(self.model.emission_distn.sigma,
                                         axis=0)

        _, filtered_mus, filtered_sigmas = \
            pylds.lds_messages_interface.kalman_filter(
                s.mu_init, s.sigma_init,
                s.A, s.B, s.sigma_states,
                s.C, s.D, s.sigma_obs,
                s.inputs, s.data)

        # We take the last input in "s", and use it to get the distribution
        # of z_0 for the forward samples.
        init_mu = s.A.dot(filtered_mus[-1]) + s.B.dot(s.inputs[-1])
        init_sigma = s.sigma_states + s.A.dot(filtered_sigmas[-1]).dot(s.A.T)

        randseq = np.zeros((Tpred - 1, s.D_latent))
        if states_noise:
            L = np.linalg.cholesky(s.sigma_states)
            randseq += np.random.randn(Tpred - 1, s.D_latent).dot(L.T)

        states = np.empty((Tpred, s.D_latent))
        obs = np.empty((Tpred, s.D_emission))
        running_x = np.concatenate((x, obs), axis=0)
        T = x.shape[0]

        if states_noise:
            states[0] = np.random.multivariate_normal(init_mu, init_sigma)
        else:
            states[0] = init_mu

        u_t_min_1 = s.inputs[-1, :]

        L = np.linalg.cholesky(s.sigma_obs)
        obs[0] = states[0].dot(s.C.T) + u_t_min_1.dot(s.D.T)
        if obs_noise:
            obs[0] += np.random.randn(1, s.D_emission).dot(L.T).flatten()

        running_x[T] = obs[0]
        # print(running_x.shape, running_x)

        if future_inputs is None:
            if dynamic_bias is None:
                future_inputs = np.zeros((Tpred, s.D_input - 1))
            else:
                future_inputs = np.zeros(
                    (Tpred, s.D_input - dynamic_bias.dim - 1))

        if self.D_input_gamma > 0:
            future_inputs = self._scale_input_by_gamma(
                future_inputs, gamma, future_dynamic_bias=dynamic_bias)

        for t in range(1, Tpred):
            u_t_min_1 = future_inputs[t - 1]

            if dynamic_bias is not None:
                db = dynamic_bias.get_last_dynamic_bias(running_x[0:T + t, :],
                                                        country_df)
                u_t_min_1 = np.append(u_t_min_1, db)

            states[t] = (s.A.dot(states[t - 1]) +
                         s.B.dot(u_t_min_1) +
                         randseq[t - 1])

            obs[t] = states[t].dot(s.C.T) + u_t_min_1.dot(s.D.T)
            if obs_noise:
                obs[t] += np.random.randn(1, s.D_emission).dot(L.T).flatten()

            running_x[T + t] = obs[t]

        return smoothed_x_mean, smoothed_x_cov, obs

    def _resample_dynamics_identity_control(self, data):

        def _empty_statistics(D_in, D_out):
            return np.array([np.zeros((D_out, D_out)),
                             np.zeros((D_out, D_in)),
                             np.zeros((D_in, D_in)), 0])

        stats = sum((self.model.dynamics_distn._get_statistics(d)
                     for d in data),
                    _empty_statistics(self.D_input, self.D_latent))

        # TODO. The prior parameters should be taken from the model,
        # not restated here.
        natural_hypparam = self.model.dynamics_distn._standard_to_natural(
            self.D_latent + 1,  # nu_0
            self.D_latent * np.eye(self.D_latent),  # S_0
            np.zeros((self.D_latent, self.D_input)),  # M_0
            self.D_latent * np.eye(self.D_input)  # K_0
        )
        mean_params = self.model.dynamics_distn._natural_to_standard(
            natural_hypparam + stats)

        A, sigma = sample_mniw(*mean_params)

        self.model.dynamics_distn.sigma = sigma
        self.model.dynamics_distn.A = np.concatenate(
            (np.eye(self.D_latent), A), axis=1)
        self.model.dynamics_distn._initialize_mean_field()

    def _resample_gamma(self):
        for i, (s, inputs) in enumerate(zip(self.model.states_list,
                                            self.inputs)):
            inputs = inputs[:-1]

            # We break the inputs into [inputs1, inputs2], where inputs1
            # correspond to the dimensions that are rescaled by gamma.
            # -- Precision --
            inputs1 = inputs[:, :self.D_input_gamma]
            B1 = self.model.B[:, :self.D_input_gamma]
            i1 = np.matmul(B1, inputs1.T)

            y = np.matmul(np.linalg.inv(self.model.sigma_states), i1)
            precision = np.sum(y * i1) + self.gamma_prior_precision

            # -- Mean times precision --
            inputs2 = inputs[:, self.D_input_gamma:]
            B2 = self.model.B[:, self.D_input_gamma:]
            i2 = np.matmul(B2, inputs2.T)

            Az = np.matmul(self.model.A, s.gaussian_states[:-1].T)
            y = s.gaussian_states[1:].T - Az - i2
            y = np.matmul(np.linalg.inv(self.model.sigma_states), y)
            mean_times_precision = np.sum(
                y * i1) + self.gamma_prior_mean_times_precision

            mu = mean_times_precision / precision
            sigma = 1 / np.sqrt(precision)
            gamma = np.random.normal(loc=mu, scale=sigma)

            self.gammas[i] = gamma

    def resample_model(
            self,
            identity_transition_matrix: bool=False,
            fixed_emission_matrix: bool=False) -> Tuple['MCMCSample', float]:

        # 1.  Resample the parameters
        # 1.1 Resample the scalar gamma for each time series' external control
        if self.D_input_gamma > 0:
            self._resample_gamma()
            self._scale_inputs_by_gamma()

        # 1.2 Resample the dynamics distribution
        if identity_transition_matrix:
            data = [np.hstack((s.inputs[:-1],
                               s.gaussian_states[1:] - s.gaussian_states[:-1])
                              )
                    for s in self.model.states_list]

            self._resample_dynamics_identity_control(data)
        else:
            self.model.resample_dynamics_distn()

        # 1.3 Resample the emission distribution
        xys = [(np.hstack((s.gaussian_states, 0 * s.inputs)),
                s.data)
               for s in self.model.states_list]

        self.model.emission_distn.resample(data=xys)

        self.model.D = np.zeros((self.D_obs, self.D_input))

        if fixed_emission_matrix:
            # Comment. In pybasicbayes's sampler for Multivariate Normal
            # Inverse Wishart, sample_mniw(...), Sigma is first sampled,
            # and then A conditioned on Sigma. It is therefore safe to reset
            #  C to C_constant. We therefore just keep the emission noise
            # sample.
            self.model.C = self.C_constant

        # 2.  Resample the states
        self.model.resample_states()

        sample = MCMCSample(A=self.model.A, B=self.model.B,
                            C=self.model.C, D=self.model.D,
                            Q=self.model.dynamics_distn.sigma,
                            R=self.model.emission_distn.sigma,
                            gamma=np.array(self.gammas))

        return sample, self.model.log_likelihood()

    # Assuming the initial parameters are decent, first sample states.
    def resample_states(self):
        self.model.resample_states()

    def _scale_input_by_gamma(self,
                              inputs: np.ndarray,
                              gamma: float,
                              future_dynamic_bias: AbstractDynamicBias=None):

        dim = 0 if future_dynamic_bias is None else future_dynamic_bias.dim

        gamma_mask = np.concatenate(
            (
                gamma * np.ones((1, self.D_input_gamma)),
                np.ones((1, self.D_input - self.D_input_gamma - dim))
            ),
            axis=1)
        return inputs * gamma_mask

    def _scale_inputs_by_gamma(self):
        for s, inputs, gamma in zip(self.model.states_list,
                                    self.inputs,
                                    self.gammas):
            s.inputs = self._scale_input_by_gamma(inputs, gamma)


class MCMCSample(object):
    def __init__(self, A, B, C, D, Q, R, gamma):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.gamma = gamma

    def __str__(self):
        string = 'A:\n' + str(self.A) + '\nB:\n' + str(
            self.B) + '\nC:\n' + str(self.C) + '\nD:\n' + str(
            self.D) + '\nQ:\n' + str(self.Q) + '\nR:\n' + str(
            self.R) + '\ngamma:\n' + str(self.gamma)
        return string
