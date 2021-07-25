import torch

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import pyro.contrib.bnn as bnn


class GenericDNN(torch.nn.Module):
    """
    Simple, generic Deep Neural Network (DNN) with one input layer, one
    hidden layer and an output layer, i.e. satisfies the criteria of being
    a 'deep' network in the universal approximation theorem (UAT) sense.
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
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        z = self.softplus(self.fc1(x)) + x  # skip-connect
        return self.fc2(z)


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
            input_size + 1,
            hidden_size + 1,
            output_size,
        )

    def forward(self, X, W1, W2):
        """
        Args:
            X: input
            W1: weights for first dense layer
            W2: weights for second dense layer
        """
        if W1.shape[-1] != X.shape[-1]:
            X = torch.cat([X, torch.ones(X.shape[0], 1)], axis=-1)

        B = torch.nn.functional.softplus(
            W1 @ X[..., None]
        )  # [b,wout,win] @ [b,win,1] -> [b,wout,1]
        B = B + X[..., None]  # skip-connect
        F = W2 @ B  # [b,wout2,wout1] @ [b,wout1,1] -> [b,wout2,1]
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
        W1 = W1.reshape(self.hidden_size, self.input_size)
        W2 = W2.reshape(self.output_size, self.hidden_size)
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

        # add bias
        X = torch.cat([X, torch.ones(batch_size, 1)], axis=-1)

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        W1_loc = torch.zeros((self.input_size * self.hidden_size), **options)
        W1_cov = torch.eye((self.input_size * self.hidden_size), **options)

        # prior for second weight matrix W2, i.e. p(W2) = MN(0,I,I)
        W2_loc = torch.zeros((self.hidden_size * self.output_size), **options)
        W2_cov = torch.eye((self.hidden_size * self.output_size), **options)

        with pyro.plate("data", batch_size):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            W1 = pyro.sample("W1", dist.MultivariateNormal(W1_loc, W1_cov))
            W2 = pyro.sample("W2", dist.MultivariateNormal(W2_loc, W2_cov))
            # undo vec(W1), i.e. make matrix
            W1 = W1.reshape(
                batch_size, self.hidden_size, self.input_size
            )  # pseudo-W1.T
            W2 = W2.reshape(
                batch_size, self.output_size, self.hidden_size
            )  # pseudo-W2.T

            # The likelihood. (equivalent to MSE-loss?)
            out = pyro.sample("y", dist.Normal(self.forward(X, W1, W2), scale=1), obs=y)

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

        # add bias
        X = torch.cat([X, torch.ones(batch_size, 1)], axis=-1)

        mu_W1_init = torch.randn(self.input_size * self.hidden_size) * 0.01
        mu_W2_init = torch.randn(self.hidden_size * self.output_size) * 0.01

        U1 = (
            torch.eye(self.input_size, **options)
            + torch.rand((self.input_size, self.input_size)) * 0.01
        )
        V1 = (
            torch.eye(self.hidden_size, **options)
            + torch.rand((self.hidden_size, self.hidden_size)) * 0.01
        )
        U2 = (
            torch.eye(self.hidden_size, **options)
            + torch.rand((self.hidden_size, self.hidden_size)) * 0.01
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

        with pyro.plate("data", batch_size):
            # W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, cov_W1))
            W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, cov1))
            W2 = pyro.sample("W2", dist.MultivariateNormal(mu_W2, cov2))


class MND_BNN_alternative(torch.nn.Module):
    """
    Identical to "MND_BNN", except using cholesky factorized covariance
    for the MultiVN distributions, in contrast to the kronicker product
    transformation from MatrixVN to MultiVN.

    This might be a bit faster.
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
            input_size + 1,
            hidden_size + 1,
            output_size,
        )

    def forward(self, X, W1, W2):
        """
        Args:
            X: input
            W1: weights for first dense layer
            W2: weights for second dense layer
        """
        if W1.shape[1] != X.shape[-1]:
            X = torch.cat([X, torch.ones(X.shape[0], 1)], axis=-1)

        B = torch.nn.functional.softplus(
            W1 @ X[..., None]
        )  # [b,wout,win] @ [b,win,1] -> [b,wout,1]
        B = B + X[..., None]  # skip-connect
        F = W2 @ B  # [b,wout2,wout1] @ [b,wout1,1] -> [b,wout2,1]
        return F[..., 0]  # [b,wout2,1] -> [b,wout2]

    def sample_weights(self):
        """
        Sample weight from variational posterior
        """
        mu1, tril_cov1, mu2, tril_cov2 = pyro.get_param_store().values()
        W1 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu1, scale_tril=tril_cov1
        ).sample()
        W2 = torch.distributions.multivariate_normal.MultivariateNormal(
            mu2, scale_tril=tril_cov2
        ).sample()
        W1 = W1.reshape(self.hidden_size, self.input_size)
        W2 = W2.reshape(self.output_size, self.hidden_size)
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

        # add bias
        X = torch.cat([X, torch.ones(batch_size, 1)], axis=-1)

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        W1_loc = torch.zeros((self.input_size * self.hidden_size), **options)
        W1_cov = torch.eye((self.input_size * self.hidden_size), **options)

        # prior for second weight matrix W2, i.e. p(W2) = MN(0,I,I)
        W2_loc = torch.zeros((self.hidden_size * self.output_size), **options)
        W2_cov = torch.eye((self.hidden_size * self.output_size), **options)

        with pyro.plate("data", batch_size):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            W1 = pyro.sample("W1", dist.MultivariateNormal(W1_loc, W1_cov))
            W2 = pyro.sample("W2", dist.MultivariateNormal(W2_loc, W2_cov))
            # undo vec(W1), i.e. make matrix
            W1 = W1.reshape(
                batch_size, self.hidden_size, self.input_size
            )  # pseudo-W1.T
            W2 = W2.reshape(
                batch_size, self.output_size, self.hidden_size
            )  # pseudo-W2.T

            # The likelihood. (equivalent to MSE-loss?)
            out = pyro.sample("y", dist.Normal(self.forward(X, W1, W2), scale=1), obs=y)

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

        # add bias
        X = torch.cat([X, torch.ones(batch_size, 1)], axis=-1)

        mu_W1_init = torch.randn(self.input_size * self.hidden_size) * 0.01
        mu_W2_init = torch.randn(self.hidden_size * self.output_size) * 0.01
        var_W1_init = torch.eye(
            (self.input_size * self.hidden_size), **options
        )  # torch.rand((self.input_size * self.hidden_size, self.input_size * self.hidden_size))
        var_W2_init = torch.eye(
            (self.hidden_size * self.output_size), **options
        )  # torch.rand((self.hidden_size * self.output_size, self.hidden_size * self.output_size))

        tril_cov1 = torch.randn(
            (self.input_size * self.hidden_size, self.input_size * self.hidden_size)
        )
        tril_cov1 = torch.tril(tril_cov1, diagonal=-1) + var_W1_init
        tril_cov2 = torch.randn(
            (self.hidden_size * self.output_size, self.hidden_size * self.output_size)
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

        with pyro.plate("data", batch_size):
            # W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, cov_W1))
            W1 = pyro.sample("W1", dist.MultivariateNormal(mu_W1, scale_tril=tril_cov1))
            W2 = pyro.sample("W2", dist.MultivariateNormal(mu_W2, scale_tril=tril_cov2))


class MND_BNN_nativehidden(torch.nn.Module):
    """
    Tries to use pyro native HiddenLayer

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
            input_size + 1,
            hidden_size + 1,
            output_size,
        )

    def forward(self, X, W1, W2):
        """
        Args:
            W1: weights for first dense layer
            W2: weights for second dense layer
        """
        B = torch.nn.functional.softplus(
            W1 @ X[..., None]
        )  # [b,wout,win] @ [b,win,1] -> [b,wout,1]
        B = B + X[..., None]
        F = W2 @ B  # [b,wout2,wout1] @ [b,wout1,1] -> [b,wout2,1]
        return F[..., 0]  # [b,wout2,1] -> [b,wout2]

    def sample_weights(self):
        a1_mean, a1_scale, a2_mean, a2_scale = pyro.get_param_store().values()
        W1 = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.flatten(a1_mean), torch.diag(torch.flatten(a1_scale))
        ).sample()
        W2 = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.flatten(a2_mean), torch.diag(torch.flatten(a2_scale))
        ).sample()
        W1 = W1.reshape(self.hidden_size, self.input_size)
        W2 = W2.reshape(self.output_size, self.hidden_size)
        return W1, W2

    def model(self, X, y):
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]

        # prior for first weight matrix W1, i.e. p(W1) = MN(0,I,I)
        a1_mean = torch.zeros((self.input_size, self.hidden_size), **options)
        a1_scale = torch.ones((self.input_size, self.hidden_size), **options)

        a2_mean = torch.zeros((self.hidden_size + 1, self.output_size), **options)
        a2_scale = torch.ones((self.hidden_size + 1, self.output_size), **options)

        # Mark batched calculations to be conditionally independent given parameters using `plate`
        with pyro.plate("data", size=batch_size):
            # Sample first hidden layer
            h1 = pyro.sample(
                "h1",
                bnn.HiddenLayer(
                    X, a1_mean, a1_scale, non_linearity=torch.nn.functional.softplus
                ),
            )
            # Sample second hidden layer
            h2 = pyro.sample(
                "h2", bnn.HiddenLayer(h1, a2_mean, a2_scale, non_linearity=lambda x: x)
            )
            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            return pyro.sample("y", dist.Normal(h2[:, 0:1], scale=1), obs=y)

    def guide(self, X, y):
        options = dict(dtype=y.dtype, device=y.device)
        batch_size = X.shape[0]
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param(
            "a1_mean", 0.01 * torch.randn((self.input_size, self.hidden_size))
        )
        a1_scale = pyro.param(
            "a1_scale",
            0.1 * torch.ones((self.input_size, self.hidden_size)),
            constraint=constraints.greater_than(0.01),
        )
        a2_mean = pyro.param(
            "a2_mean", 0.01 * torch.randn((self.hidden_size + 1, self.output_size))
        )
        a2_scale = pyro.param(
            "a2_scale",
            0.1 * torch.ones((self.hidden_size + 1, self.output_size)),
            constraint=constraints.greater_than(0.01),
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
