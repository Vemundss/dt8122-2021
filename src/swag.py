"""
Implementation of the SWAG-algorithm, as described in the paper:
'A Simple Baseline for Bayesian Uncertainty in Deep Learning' J. Maddox et al. (2019)
"""
import numpy as np
import torch
import tqdm

import sys
import os
import copy
import operator
from pathlib import Path

from methods import *


def generic_train_loop(net, trainloader, optimizer, criterion, nepochs):
    """
    Classic DL training loop
    """
    loss_history = []
    # loop over the dataset many times (use tqdm, i.e. progress bar, if many epochs)
    for epoch in tqdm.trange(nepochs) if nepochs > 10 else range(nepochs):

        # generic torch training loop
        running_loss = 0.0
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
        loss_history.append(running_loss / len(trainloader))

    return net, loss_history


def train_swag(
    net,
    trainloader,
    optimizer,
    criterion,
    init_epochs=100,
    sampling_epochs=5,
    nsamples=20,
    path_to_checkpoints="../checkpoints/",
    custom_addition_name="",
):
    """
    Training phase of a SWAG-based model

    Args:
        net: (torch.nn.Module) model
        trainloader: (torch.utils.data.DataLoader) training data loader
        optimizer: (torch.optim) optimizer
        criterion: (torch.nn) loss function, e.g. MSELoss()
        init_epochs: (int) number of epochs for initial training ("warm-up")
        sampling_epochs: (int) number of training epochs before sampling weights
        nsamples: (int) total number of weights to sample
        path_to_checkpoints: (str) path to checkpoint folder to save weights
        custom_addition_name: (str) extra trailing naming to checkpoint folder
    """
    # ---------
    # Vanilla training of a newly instantiated model
    # ---------
    print("Training to initial weight-solution (MLE)")
    net, loss_history = generic_train_loop(
        net, trainloader, optimizer, criterion, nepochs=init_epochs
    )

    # Make dirs and save checkpoint
    Path(path_to_checkpoints + trainloader.dataset.name + custom_addition_name).mkdir(
        parents=True, exist_ok=True
    )
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "loss_history": loss_history,
            "epoch": init_epochs,
        },
        path_to_checkpoints
        + trainloader.dataset.name
        + custom_addition_name
        + f"/{init_epochs:04d}",
    )

    # ---------
    # Continue training and saving (nsamples) new weights from initial weights
    # ---------
    print(
        "Iterate/train further from initial weight-solution ('Sample' around posterior mode)"
    )
    for weight_sample in tqdm.trange(nsamples):
        net, loss_history = generic_train_loop(
            net, trainloader, optimizer, criterion, nepochs=sampling_epochs
        )
        current_epoch = init_epochs + weight_sample * sampling_epochs
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "loss_history": loss_history,
                "epoch": current_epoch,
            },
            path_to_checkpoints
            + trainloader.dataset.name
            + custom_addition_name
            + f"/{current_epoch:04d}",
        )


def inference_swag(name, checkpoint_path="../checkpoints/"):
    """
    Infer approximate weight sample mean and covariance.

    Note! This method loads all disc-stored weights into RAM, hence
    this obviously assumes not too large models - in terms of weights, nor
    too many checkpoints.

    Args:
        name: (str) common checkpoint name, e.g. 'yacht'
        checkpoint_path: (str) path to checkpoints
    """
    tweights = []
    onlyfiles = [
        f
        for f in os.listdir(checkpoint_path + name + "/")
        if os.path.isfile(os.path.join(checkpoint_path + name + "/", f))
    ]
    for file in onlyfiles:
        odict = torch.load(checkpoint_path + name + "/" + file)["model_state_dict"]
        tweight = ordereddict2tensor(odict)
        tweights.append(tweight)

    if len(tweights) == 0:
        raise Exception(
            f"Custom Exception! The weights at <{checkpoint_path + name}> does not exist. TRAIN first!"
        ) from None

    # do inference
    tweights = torch.stack(tweights)
    theta_SWA = torch.mean(tweights, axis=0)
    theta_second_moment = torch.mean(tweights ** 2, axis=0)
    cov_diag = theta_second_moment - theta_SWA ** 2
    D = tweights - theta_SWA
    return theta_SWA, cov_diag, D


def sample_posterior_swag(theta_SWA, cov_diag, D):
    """
    Sample the inferred approximate posterior distribution found by SWAG,
    as specified by Equation (1) in their article.
    """
    z1 = torch.normal(torch.zeros(theta_SWA.shape), 1)
    z2 = torch.normal(torch.zeros(D.shape[0]), 1)

    z1_tmp = 1 / np.sqrt(2) * torch.sqrt(cov_diag) * z1
    z2_tmp = 1 / np.sqrt(2 * D.shape[0]) * D.T @ z2  # per "layer"
    posterior_sample = theta_SWA + z1_tmp + z2_tmp
    return posterior_sample


def monte_carlo_PI(x, model, theta_SWA, cov_diag, D, nsamples=50, percentile=0.95):
    """
    Calculate prediction intervals using monte carlo.

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
        tweights = sample_posterior_swag(theta_SWA, cov_diag, D)
        model.load_state_dict(tensor2ordereddict(tweights, model.state_dict()))

        # do prediction with current (wrt. weights) model
        y_preds.append(model(x).detach().numpy())
    y_preds = np.array(y_preds)

    # sort a long sample-dimension
    y_preds = np.sort(y_preds, axis=0)

    idx_percentile = round((1 - percentile) * nsamples)
    lower_pi, upper_pi = y_preds[idx_percentile], y_preds[-idx_percentile]

    # mean (wrt. weights) model
    model.load_state_dict(tensor2ordereddict(theta_SWA, model.state_dict()))
    predictive_swa = model(x).detach().numpy()

    # mean (wrt. the approximate posterior predictive distribution) model
    predictive_mean = np.mean(y_preds, axis=0)

    return lower_pi, upper_pi, predictive_swa, predictive_mean
