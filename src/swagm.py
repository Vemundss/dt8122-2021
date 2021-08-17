"""
Implementation of the SWAGM-algorithm, a novel extension to the SWAG-algorithm:
'A Simple Baseline for Bayesian Uncertainty in Deep Learning' J. Maddox et al. (2019)

SWAGM => SWAG-mixtures

Specifically the novelty can be summarized as:
1. Extend posterior functional form from a multivariate normal distribution
   to a mixture of multivariate normal distributions
2. The mixture is created by running SWAG multiple times.
3. Each sample from a new mixture (new run of SWAG) is then checked to see whether
   it is likely under any of the other mixtures. If so, then it is added to an 
   existing mixture sample. Otherwise, it defines a new mixture.
"""
import numpy as np
import torch
import copy
from scipy.spatial.distance import cdist
from glob import glob
import shutil
import os

from swag import *
from methods import *


def Hotelling_T_squared():
    """
    UNUSED

    Test similarity between two distributions
    """
    return None


def KLdiv(X, logP, logQ):
    """
    UNUSED

    Compute the Kullback-Leibler divergence between P and Q, respectively,
    for the data X. Q is sometimes considered as the approximation of P.

    function should be of type:
    torch.distributions.distribution.Distribution.log_prob

    Args:
        X, torch tensor (nsamples, features): data
        logP, function: log(P(X)) i.e. log probability of the P-distribution
        logQ, function: log(Q(X)) i.e. log probability of the Q-distribution

    Returns:
        KL divergence: KL(P || Q)
    """
    return torch.sum(torch.exp(logP(X)) * (logP(X) - logQ(X)))


def gaussian_kernel(X, Y, h=1):
    """
    Gaussian kernel statistic as described by:
    https://normaldeviate.wordpress.com/2012/07/14/modern-two-sample-tests/

    Args:
        X, torch tensor (nsamples, features): data
        Y, torch tensor (nsamples, features): data
        h: tuning parameter for the gaussian kernel
    """
    return torch.exp(-torch.cdist(X, Y) / h ** 2)


def kernel_test_statistic(X, Y, kernel=None):
    """
    Kernel statistic as described by:
    https://normaldeviate.wordpress.com/2012/07/14/modern-two-sample-tests/

    Args:
        X, torch tensor (nsamples, features): data
        Y, torch tensor (nsamples, features): data
        kernel, function(X,Y)->scalar: a kernel function
    """
    if kernel is None:
        # default kernel
        kernel = gaussian_kernel

    return (
        torch.mean(kernel(X, X))
        - 2 * torch.mean(kernel(X, Y))
        + torch.mean(kernel(Y, Y))
    )


def permutation_test(X, Y, test_statistic, nsamples=1000):
    """
    Test whether two samples (X and Y) are sampled from the same distribution
    using a permutation test.

    Excellent explanations/resources of permutation testing (fetched 14/08-2021):
    https://normaldeviate.wordpress.com/2012/07/14/modern-two-sample-tests/
    https://www.jwilber.me/permutationtest/

    Args:
        X, torch tensor (nsamples, features): data
        Y, torch tensor (nsamples, features): data
        test_statistic, function(X,Y)->scalar: a test statistic
        nsamples: #permutations to calculate p-value for
    """
    XY_stat = test_statistic(X, Y)
    stats = torch.zeros(nsamples)
    XY = torch.cat([X, Y])
    # print(XY_stat)
    for ni in range(nsamples):
        perm_idxs = torch.randperm(XY.shape[0])
        # mutually exclusive (every other) sampled indices
        stat = test_statistic(XY[perm_idxs[::2]], XY[perm_idxs[1::2]])
        stats[ni] = stat

    # print(np.sort(np.around(stats.detach().numpy(),decimals=3)))

    # p_value = torch.sum(XY_stat >= stats) / nsamples
    p_value = torch.sum(stats >= XY_stat) / nsamples
    return p_value


def get_swag_weights_as_tensor(path_to_weights):
    """
    Helper-function for loading a matrix of weights from one mixture,
    i.e. from one SWAG-run.
    """
    tweights = []
    onlyfiles = [
        f
        for f in os.listdir(path_to_weights)
        if os.path.isfile(os.path.join(path_to_weights, f))
    ]
    for file in onlyfiles:
        odict = torch.load(path_to_weights + file)["model_state_dict"]
        tweight = ordereddict2tensor(odict)
        tweights.append(tweight)

    return torch.stack(tweights)


def collect_similar_modes(
    name,
    nsamples=1000,
    alpha=0.05,
    correct_alpha=True,
    checkpoint_path="../checkpoints/",
):
    """
    Novelty! Check whether two sets of samples are similar enough
    to be sampled from the same data generating process (distribution)

    Args:
        name (str): name of dataset / general name for model
        nsamples (int): #permutations to base p-value on. More is better..
                        but expensive
        alpha (float): significance level
    """
    path_to_swagm_weights = checkpoint_path + name + "-swagm/"
    mixture_weight_names = [
        os.path.basename(x) for x in glob(path_to_swagm_weights + "*")
    ]

    if correct_alpha:
        alpha_corrected = alpha / (
            len(mixture_weight_names) ** 2 - len(mixture_weight_names)
        )
        print(f"ALPHA_corrected is alpha/(n**2 - n) = {alpha_corrected}")
        print("The null-hypothesis H_0 is: Samples come from the same distribution")
    else:
        print(f"{alpha=}")

    for mixture_weight_name1 in mixture_weight_names:
        tweights1 = get_swag_weights_as_tensor(
            path_to_swagm_weights + mixture_weight_name1 + "/"
        )

        for mixture_weight_name2 in mixture_weight_names:
            if mixture_weight_name1 == mixture_weight_name2:
                # no need to compare a sample with itself
                continue
            tweights2 = get_swag_weights_as_tensor(
                path_to_swagm_weights + mixture_weight_name2 + "/"
            )

            # Check whether samples are from the same distribution
            p_value = permutation_test(
                X=tweights1,
                Y=tweights2,
                test_statistic=kernel_test_statistic,
                nsamples=nsamples,
            )
            if p_value < alpha_corrected:
                print(
                    f"{p_value=} for samples: {mixture_weight_name1} and {mixture_weight_name2}. "
                    + "Thus, <<<H0 is FALSE>>>. Continuing."
                )

            else:
                print(
                    f"{p_value=} for samples: {mixture_weight_name1} and {mixture_weight_name2}. "
                    + "Thus, <<<H0 is TRUE>>>. Collecting mixtures."
                )
                # samples come from same distribution - collect weights in same folder,
                # which also increases samples available for that mixture
                for weight_file_name in os.listdir(
                    path_to_swagm_weights + mixture_weight_name2 + "/"
                ):
                    shutil.move(
                        path_to_swagm_weights
                        + mixture_weight_name2
                        + "/"
                        + weight_file_name,
                        path_to_swagm_weights
                        + mixture_weight_name1
                        + "/"
                        + mixture_weight_name2
                        + "_"
                        + weight_file_name,
                    )

                # delete folder which contained samples before
                shutil.rmtree(path_to_swagm_weights + mixture_weight_name2)
                # remove name of that folder from lists being looped
                mixture_weight_names.remove(mixture_weight_name2)


def train_swagm(init_net, nmixtures, **kwargs):
    """
    Args:
        DNN_class (torch.nn.Module): class (NOT object)
        nmixtures (int): number of times to run the SWAG algorithm -
                         upper bound on number of mixtures
        args*, **kwargs: args and kwargs for train_swag(), exlcuding
                         'custom_addition_name'
    """
    print(f"TRAINING SWAGM using SWAG {nmixtures=} times")
    for i in range(nmixtures):
        net, criterion, optimizer = init_net()
        train_swag(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            custom_addition_name=f"-swagm/swagm{i:03d}",
            **kwargs,
        )


def inference_swagm(name, checkpoint_path="../checkpoints/"):
    """
    Do SWAG inference. Also infer mixture distribution.

    Here, the mixture distribution is a Categorical distribution with 'nmixtures'
    number of categories. The probability of sampling from a mixture shall depend
    on the fraction of samples within that mixture.
    """
    path_to_swagm_weights = checkpoint_path + name + "-swagm/"
    mixture_weight_names = [
        os.path.basename(x) for x in glob(path_to_swagm_weights + "*")
    ]
    mixture_weight_names.sort()

    nmixtures = len(mixture_weight_names)
    ns = torch.zeros(nmixtures)
    theta_SWAs, cov_diags, Ds = [], [], []
    # for efficiency in this project we load and keep all distribution
    # quantities (low-rank covariances and means) in memory.
    # This could partially be reduced (at the expense of CPU resources)
    # by only loading one mixture's quantities depending on the categorical
    # sample. However, when we want to sample multiple times, we would have to
    # reload each distributions quantities for (almost) every new sample.
    # Instead we load all the quantities, so that sampling is efficient (wrt. CPU)
    # as long as we have the memory capacity for it.
    for i, mixture_weight_name in enumerate(mixture_weight_names):
        onlyfiles = [
            f
            for f in os.listdir(path_to_swagm_weights + mixture_weight_name + "/")
            if os.path.isfile(
                os.path.join(path_to_swagm_weights + mixture_weight_name + "/", f)
            )
        ]
        ns[i] = len(onlyfiles)

        # do SWAG inference on each mixture
        theta_SWA, cov_diag, D = inference_swag(
            name + "-swagm/" + mixture_weight_name, checkpoint_path="../checkpoints/"
        )
        theta_SWAs.append(theta_SWA)
        cov_diags.append(cov_diag)
        Ds.append(D)

    N = torch.sum(ns)
    probs = ns / N
    catdist = torch.distributions.categorical.Categorical(probs)

    return theta_SWAs, cov_diags, Ds, catdist


def sample_posterior_swagm(theta_SWAs, cov_diags, Ds, catdist):
    """
    Sample from categorical distribution followed by sampling SWAG for that
    mixture
    """
    mixture = catdist.sample()
    theta_SWA, cov_diag, D = theta_SWAs[mixture], cov_diags[mixture], Ds[mixture]

    z1 = torch.normal(torch.zeros(theta_SWA.shape), 1)
    z2 = torch.normal(torch.zeros(D.shape[0]), 1)

    z1_tmp = 1 / np.sqrt(2) * torch.sqrt(cov_diag) * z1
    z2_tmp = 1 / np.sqrt(2 * D.shape[0]) * D.T @ z2  # per "layer"
    posterior_sample = theta_SWA + z1_tmp + z2_tmp
    return posterior_sample


def monte_carlo_PI_swagm(
    x, model, theta_SWAs, cov_diags, Ds, catdist, nsamples=50, percentile=0.95
):
    """
    Calculate prediction intervals for SWAGM using monte carlo.

    The method can be summarized as:
        - Sample weights
        - Do predictions: y_preds
        - Sort predictions
        - Choose predictions at percentile-indices
    """
    # Forward x nsamples times with different sampled weights
    y_preds = []
    for i in range(nsamples):
        # sample weights and add those weights to the model
        tweights = sample_posterior_swagm(theta_SWAs, cov_diags, Ds, catdist)
        model.load_state_dict(tensor2ordereddict(tweights, model.state_dict()))

        # do prediction with current (wrt. weights) model
        y_preds.append(model(x).detach().numpy())
    y_preds = np.array(y_preds)

    # sort a long sample-dimension
    y_preds = np.sort(y_preds, axis=0)

    idx_percentile = round((1 - percentile) * nsamples)
    lower_pi, upper_pi = y_preds[idx_percentile], y_preds[-idx_percentile]

    # mean (wrt. the approximate posterior predictive distribution) model
    predictive_mean = np.mean(y_preds, axis=0)

    return lower_pi, upper_pi, predictive_mean
