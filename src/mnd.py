"""
Custom implementation of the variational algorithm described in paper: 
'Multiplicative Normalizing Flows for Variational Bayesian Neural Networks' 
by Louizos and Welling (2017)
"""
import sys
import torch
import numpy as np

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import pyro.contrib.bnn as bnn
from pyro.contrib.bnn.utils import adjoin_ones_vector


class MND_BNN(torch.nn.Module):
    """
    Based on the paper:

    "Structured and Efficient Variational Deep Learning with
    Matrix Gaussian Posteriors" - Louizos and Welling
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
            input_size: shape of input - excluding batch_size
            hidden_size: shape of hidden - excluding batch_size
            output_size: shape of output - excluding batch_size
        """
        super().__init__()
        # add 1 to include bias
        self.input_size, self.hidden_size, self.output_size = (
            input_size,
            hidden_size,
            output_size,
        )

    def forward(self, X, W1, W2):
        """
        Args:
            X: input
            W1: weights for first dense layer
            W2: weights for second dense layer
        """
        B = torch.nn.functional.softplus(
            W1 @ adjoin_ones_vector(X)[..., None]  # add bias
        )  # [b,wout,win] @ [b,win,1] -> [b,wout,1]
        B = B + X[..., None]  # skip-connect
        F = (
            W2 @ adjoin_ones_vector(B[..., 0])[..., None]
        )  # [b,wout2,wout1] @ [b,wout1,1] -> [b,wout2,1]
        return F[..., 0]  # [b,wout2,1] -> [b,wout2]

    def sample_weights(self):
        """
        Sample weight from variational posterior
        """
        mu1, mu2, U1, V1, U2, V2 = pyro.get_param_store().values()
        cov1 = torch.kron(V1, U1)
        cov2 = torch.kron(V2, U2)

        # Sample matrix variate gaussian from multivariate gaussian
        # The relation between MatrixVG and MultiVG can be found from:
        # https://en.wikipedia.org/wiki/Matrix_normal_distribution
        W1 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu1, cov1
        ).sample()
        W2 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu2, cov2
        ).sample()

        # reshape to matrices (inverse of vec(matrix))
        W1 = W1.reshape(self.hidden_size, self.input_size + 1)
        W2 = W2.reshape(self.output_size, self.hidden_size + 1)
        return W1, W2

    def model(self, X, y):
        """
        Defines the model, i.e. the prior and the likelihood

        Args:
            X (batch_size, N): Explanatory variables, i.e. input
            y (batch_size): Target variable, i.e. output
        """
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        W1_loc = torch.zeros(((self.input_size + 1) * self.hidden_size), **options)
        W1_cov = torch.eye(((self.input_size + 1) * self.hidden_size), **options)

        # prior for second weight matrix W2, i.e. p(W2) = MN(0,I,I)
        W2_loc = torch.zeros(((self.hidden_size + 1) * self.output_size), **options)
        W2_cov = torch.eye(((self.hidden_size + 1) * self.output_size), **options)

        # tau = pyro.sample("tau", dist.Gamma(6,6))

        with pyro.plate("data", batch_size):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            W1 = pyro.sample("W1", dist.MultivariateNormal(W1_loc, W1_cov))
            W2 = pyro.sample("W2", dist.MultivariateNormal(W2_loc, W2_cov))
            # undo vec(W1), i.e. make matrix

            W1 = W1.reshape(
                batch_size, self.hidden_size, self.input_size + 1
            )  # pseudo-W1.T
            W2 = W2.reshape(
                batch_size, self.output_size, self.hidden_size + 1
            )  # pseudo-W2.T

            # The likelihood. (equivalent to MSE-loss?)
            F = self.forward(X, W1, W2)
            out = pyro.sample("y", dist.Normal(F, scale=1.0).to_event(2), obs=y)

    def guide(self, X, y):
        """
        Variational posterior. I.e. the posterior we try to learn / make as
        similar as possible to the true posterior.

        Args:
            X (batch_size, N): Explanatory variables, i.e. input
            y (batch_size): Target variable, i.e. output
        """
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]

        mu_W1_init = torch.randn((self.input_size + 1) * self.hidden_size) * 0.01
        mu_W2_init = torch.randn((self.hidden_size + 1) * self.output_size) * 0.01

        U1 = (
            torch.eye(self.input_size + 1, **options)
            + torch.rand((self.input_size + 1, self.input_size + 1)) * 0.01
        )
        V1 = (
            torch.eye(self.hidden_size, **options)
            + torch.rand((self.hidden_size, self.hidden_size)) * 0.01
        )
        U2 = (
            torch.eye(self.hidden_size + 1, **options)
            + torch.rand((self.hidden_size + 1, self.hidden_size + 1)) * 0.01
        )
        V2 = (
            torch.eye(self.output_size, **options)
            + torch.rand((self.output_size, self.output_size)) * 0.01
        )

        mu_W1 = pyro.param("mu_W1", mu_W1_init)
        mu_W2 = pyro.param("mu_W2", mu_W2_init)
        U1 = pyro.param("U1", U1, constraint=constraints.positive)
        V1 = pyro.param("V1", V1, constraint=constraints.positive)
        U2 = pyro.param("U2", U2, constraint=constraints.positive)
        V2 = pyro.param("V2", V2, constraint=constraints.positive)

        cov1 = torch.kron(V1, U1)
        cov2 = torch.kron(V2, U2)

        # tau = pyro.sample("tau", dist.Gamma(6, 6))

        with pyro.plate("data", batch_size):
            # W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, cov_W1))
            W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, cov1))
            W2 = pyro.sample("W2", dist.MultivariateNormal(mu_W2, cov2))


class MND_BNN_alternative(torch.nn.Module):
    """
    Identical to "MND_BNN", except using cholesky factorized covariance
    for the MultiVN distributions, in contrast to the kronicker product
    transformation from MatrixVN to MultiVN.

    This is a bit faster.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
                input_size: shape of input - excluding batch_size
                hidden_size: shape of hidden - excluding batch_size
                output_size: shape of output - excluding batch_size
        """
        super().__init__()
        # Adds 1 to include bias
        self.input_size, self.hidden_size, self.output_size = (
            input_size,
            hidden_size,
            output_size,
        )

    def forward(self, X, W1, W2):
        """
        Args:
            X: input
            W1: weights for first dense layer
            W2: weights for second dense layer
        """
        B = torch.nn.functional.softplus(
            W1 @ adjoin_ones_vector(X)[..., None]  # add bias
        )  # [b,wout,win] @ [b,win,1] -> [b,wout,1]
        B = B + X[..., None]  # skip-connect
        F = (
            W2 @ adjoin_ones_vector(B[..., 0])[..., None]
        )  # [b,wout2,wout1] @ [b,wout1,1] -> [b,wout2,1]
        return F[..., 0]  # [b,wout2,1] -> [b,wout2]

    def sample_weights(self):
        """
        Sample weight from variational posterior
        """
        mu1, tril_cov1, mu2, tril_cov2, alpha, beta = pyro.get_param_store().values()
        W1 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu1, scale_tril=tril_cov1
        ).sample()
        W2 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu2, scale_tril=tril_cov2
        ).sample()
        W1 = W1.reshape(self.hidden_size, self.input_size + 1)
        W2 = W2.reshape(self.output_size, self.hidden_size + 1)
        return W1, W2

    def monte_carlo_PI(self, X, nsamples=50, percentile=0.95):
        """
        Calculate monte carlo prediction interval by sampling posterior
        and forwarding.
        """
        mu1, _, mu2, _, _, _ = pyro.get_param_store().values()
        self.sample_weights()

        y_preds = []
        for i in range(nsamples):
            W1, W2 = self.sample_weights()
            y_preds.append(
                self(X, W1, W2).detach().numpy()
            )  # forward (nsamples, mini-batch)

        y_preds = np.array(y_preds)
        y_preds = np.sort(y_preds, axis=0)
        idx_percentile = round((1 - percentile) * nsamples)
        lower_pi, upper_pi = y_preds[idx_percentile], y_preds[-idx_percentile]

        W1 = mu1.reshape(self.hidden_size, self.input_size + 1)
        W2 = mu2.reshape(self.output_size, self.hidden_size + 1)
        posterior_mean = (
            self(X, W1, W2).detach().numpy()
        )  # forward with posterior mean weights

        predictive_mean = np.mean(y_preds, axis=0)

        return lower_pi, upper_pi, posterior_mean, predictive_mean

    def model(self, X, y):
        """
        Defines the model, i.e. the prior and the likelihood

        Args:
            X (batch_size, N): Explanatory variables, i.e. input
            y (batch_size): Target variable, i.e. output
        """
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        W1_loc = torch.zeros(((self.input_size + 1) * self.hidden_size), **options)
        W1_cov = torch.eye(((self.input_size + 1) * self.hidden_size), **options)

        # prior for second weight matrix W2, i.e. p(W2) = MN(0,I,I)
        W2_loc = torch.zeros(((self.hidden_size + 1) * self.output_size), **options)
        W2_cov = torch.eye(((self.hidden_size + 1) * self.output_size), **options)

        with pyro.plate("data", batch_size):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            W1 = pyro.sample("W1", dist.MultivariateNormal(W1_loc, W1_cov))
            W2 = pyro.sample("W2", dist.MultivariateNormal(W2_loc, W2_cov))
            # undo vec(W1), i.e. make matrix
            W1 = W1.reshape(
                batch_size, self.hidden_size, self.input_size + 1
            )  # pseudo-W1.T
            W2 = W2.reshape(
                batch_size, self.output_size, self.hidden_size + 1
            )  # pseudo-W2.T

            # The likelihood. (equivalent to MSE-loss?)
            F = self.forward(X, W1, W2)
            tau = pyro.sample("tau", dist.Gamma(6.0, 6.0))
            out = pyro.sample(
                "y", dist.Normal(F, scale=1 / tau[..., None]).to_event(2), obs=y
            )

    def guide(self, X, y):
        """
        Variational posterior. I.e. the posterior we try to learn / make as
        similar as possible to the true posterior.

        Args:
            X (batch_size, N): Explanatory variables, i.e. input
            y (batch_size): Target variable, i.e. output
        """
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]

        mu_W1_init = torch.randn((self.input_size + 1) * self.hidden_size) * 0.01
        mu_W2_init = torch.randn((self.hidden_size + 1) * self.output_size) * 0.01
        var_W1_init = torch.eye(
            ((self.input_size + 1) * self.hidden_size), **options
        )  # torch.rand((self.input_size * self.hidden_size, self.input_size * self.hidden_size))
        var_W2_init = torch.eye(
            ((self.hidden_size + 1) * self.output_size), **options
        )  # torch.rand((self.hidden_size * self.output_size, self.hidden_size * self.output_size))

        tril_cov1 = torch.randn(
            (
                (self.input_size + 1) * self.hidden_size,
                (self.input_size + 1) * self.hidden_size,
            )
        )
        tril_cov1 = torch.tril(tril_cov1, diagonal=-1) + var_W1_init
        tril_cov2 = torch.randn(
            (
                (self.hidden_size + 1) * self.output_size,
                (self.hidden_size + 1) * self.output_size,
            )
        )
        tril_cov2 = torch.tril(tril_cov2, diagonal=-1) + var_W2_init

        mu_W1 = pyro.param("mu_W1", mu_W1_init)
        tril_cov1 = pyro.param(
            "tril_cov1", tril_cov1, constraint=constraints.lower_cholesky
        )
        mu_W2 = pyro.param("mu_W2", mu_W2_init)
        tril_cov2 = pyro.param(
            "tril_cov2", tril_cov2, constraint=constraints.lower_cholesky
        )

        alpha = pyro.param("alpha", torch.tensor(6.0), constraint=constraints.positive)
        beta = pyro.param("beta", torch.tensor(6.0), constraint=constraints.positive)

        with pyro.plate("data", batch_size):
            tau = pyro.sample("tau", dist.Gamma(alpha, beta))
            W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, scale_tril=tril_cov1))
            W2 = pyro.sample("W2", dist.MultivariateNormal(mu_W2, scale_tril=tril_cov2))


class HiddenLayer2(bnn.HiddenLayer):
    """
    Tries to fix HiddenLayer after it broke due to pyro-update.
    Fails...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_shape = self.X.shape[0:1]
        self._event_shape = self.A_mean.shape[1:]


class MND_BNN_nativehidden(torch.nn.Module):
    """
    Stopped working after I upgraded pyro... ridiculous

    Uses pyro native HiddenLayer

    Takes inspiration from:
    https://alsibahi.xyz/snippets/2019/06/15/pyro_mnist_bnn_kl.html
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Args:
                input_size: shape of input - excluding batch_size
                hidden_size: shape of hidden - excluding batch_size
                output_size: shape of output - excluding batch_size
        """
        super().__init__()
        self.input_size, self.hidden_size, self.output_size = (
            input_size,
            hidden_size,
            output_size,
        )

    def forward(self, X):
        # res.append(t.nodes['logits']['value'])
        t = pyro.poutine.trace(self.guide).get_trace(X, None)
        return t.nodes["h2"]["value"][:, 0]

    def sample_weights(self):
        a1_mean, a1_scale, a2_mean, a2_scale = pyro.get_param_store().values()
        W1 = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.flatten(a1_mean), torch.diag(torch.flatten(a1_scale))
        ).sample()
        W2 = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.flatten(a2_mean), torch.diag(torch.flatten(a2_scale))
        ).sample()
        W1 = W1.reshape(self.hidden_size, self.input_size)
        W2 = W2.reshape(self.output_size, self.hidden_size + 1)
        return W1, W2

    def model(self, X, y=None):
        options = dict(dtype=X.dtype, device=X.device)
        batch_size = X.shape[0]

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        a1_mean = torch.zeros((self.input_size + 1, self.hidden_size), **options)
        a1_scale = torch.ones((self.input_size + 1, self.hidden_size), **options)

        a2_mean = torch.zeros((self.hidden_size + 1, self.output_size), **options)
        a2_scale = torch.ones((self.hidden_size + 1, self.output_size), **options)

        # Conditionally independent data (mini-batch) dimension
        with pyro.plate("data", size=batch_size):
            # Sample first hidden layer

            """
            H1 = bnn.HiddenLayer(
                adjoin_ones_vector(X),  # X, including bias
                a1_mean,
                a1_scale,
                non_linearity=torch.nn.functional.softplus,
                include_hidden_bias=True,  # sadly "False" does not work..
            )
            """

            h1 = pyro.sample(
                "h1",
                HiddenLayer2(
                    adjoin_ones_vector(X),  # X, including bias
                    a1_mean,
                    a1_scale,
                    non_linearity=torch.nn.functional.softplus,
                    include_hidden_bias=True,  # sadly "False" does not work..
                ),
            )
            # Sample second hidden layer
            h1 = h1[..., :-1]  # undo 'include_hidden_bias' manually

            h2 = pyro.sample(
                "h2",
                HiddenLayer2(
                    adjoin_ones_vector(h1 + X),  # skip-connect and add bias
                    a2_mean,
                    a2_scale,
                    non_linearity=lambda x: x,
                    include_hidden_bias=True,
                ),
            )
            h2 = h2[..., :-1]  # undo 'include_hidden_bias' manually

            if y is not None:
                # Condition on observed labels, so it calculates the log-likehood loss when training using VI
                pyro.sample("y", dist.Normal(h2, scale=1), obs=y)

    def guide(self, X, y):
        options = dict(dtype=X.dtype, device=X.device)
        batch_size = X.shape[0]

        a1_mean = pyro.param(
            "a1_mean", 0.01 * torch.randn((self.input_size, self.hidden_size))
        )
        a1_scale = pyro.param(
            "a1_scale",
            0.1 * torch.ones((self.input_size, self.hidden_size)),
            constraint=constraints.positive,  # greater_than(0.001),
        )
        a2_mean = pyro.param(
            "a2_mean", 0.01 * torch.randn((self.hidden_size + 1, self.output_size))
        )
        a2_scale = pyro.param(
            "a2_scale",
            0.1 * torch.ones((self.hidden_size + 1, self.output_size)),
            constraint=constraints.positive,  # greater_than(0.001),
        )

        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate("data", size=batch_size):
            # Sample first hidden layer
            h1 = pyro.sample(
                "h1",
                bnn.HiddenLayer(
                    X, a1_mean, a1_scale, non_linearity=torch.nn.functional.softplus
                ),
            )
            # Sample second hidden layer
            # print(h1)
            h2 = pyro.sample(
                "h2", bnn.HiddenLayer(h1, a2_mean, a2_scale, non_linearity=lambda x: x)
            )
