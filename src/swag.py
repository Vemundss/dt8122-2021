"""
Implementation of the SWAG-algorithm, as described in the paper:
'A Simple Baseline for Bayesian Uncertainty in Deep Learning' J. Maddox et al.
"""
import numpy as np
import torch
import tqdm

import os
import copy
import operator

from methods import torch_weight_operation


def train_swag(
    net,
    trainloader,
    optimizer,
    criterion,
    init_epochs=100,
    sampling_epochs=5,
    nsamples=20,
    path_to_checkpoints="../checkpoints/",
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
    """
    # ---------
    # Vanilla training of a newly instantiated model
    # ---------
    print("Training to initial weight-solution (MLE)")
    loss_history = []
    for epoch in tqdm.trange(init_epochs):  # loop over the dataset multiple times

        # generic torch training loop
        running_loss = 0.0
        for i, data in enumerate(trainloader):
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

    # print("Saving initial weights")
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "loss_history": loss_history,
            "epoch": init_epochs,
        },
        path_to_checkpoints + trainloader.dataset.name + f"-{init_epochs:04d}",
    )

    # ---------
    # Continue training and saving (nsamples) new weights from initial weights
    # ---------
    print(
        "Iterate/train further from initial weight-solution ('Sample' posterior mode)"
    )
    for weight_sample in tqdm.trange(nsamples):
        loss_history = []
        for _ in range(sampling_epochs):

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

                running_loss += loss.item()
            loss_history.append(running_loss / len(trainloader))

        # print(f"Saving {weight_sample=} weights")
        current_epoch = init_epochs + weight_sample * sampling_epochs
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "loss_history": loss_history,
                "epoch": current_epoch,
            },
            path_to_checkpoints + trainloader.dataset.name + f"-{current_epoch:04d}",
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
    weights = []
    onlyfiles = [
        f
        for f in os.listdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, f))
    ]
    for file in onlyfiles:
        if name in file:
            weights.append(torch.load(checkpoint_path + file)["model_state_dict"])

    # initialise inference quantities
    theta_SWA = copy.deepcopy(weights[0])
    theta_second_moment = torch_weight_operation(
        theta_SWA, theta_SWA, operator.mul, deepcopy=True
    )
    # sum all weights
    for weight in weights[1:]:
        theta_SWA = torch_weight_operation(
            theta_SWA, weight, operator.add, deepcopy=False
        )

        weight_squared = torch_weight_operation(
            weight, weight, operator.mul, deepcopy=True
        )
        theta_second_moment = torch_weight_operation(
            theta_second_moment, weight_squared, operator.add, deepcopy=False
        )

    # rescale (i.e. mean vs. sum)
    theta_SWA = torch_weight_operation(
        theta_SWA, len(weights), operator.truediv, deepcopy=False
    )
    theta_second_moment = torch_weight_operation(
        theta_second_moment, len(weights), operator.truediv, deepcopy=False
    )

    cov_diag = torch_weight_operation(
        theta_second_moment,
        torch_weight_operation(theta_SWA, theta_SWA, operator.mul, deepcopy=True),
        operator.sub,
        deepcopy=True,
    )

    # Calculate low-rank approximate sample covariance
    # NOTE! We have access to theta_SWA, and thus use this to calculate the
    # columns of D, rather than the running average - as they do.
    # This will naturally lead to a better estimate of D - which we do not
    # complain about :) - this works because we have few explanatory variables
    # in our dataset, and because our models are relatively small (few parameters)
    D = []
    for weight in weights:
        # we do not need the RAM-loaded weights anymore, so we can overwrite them,
        # i.e., we do not need deepcopy.
        D.append(
            torch_weight_operation(weight, theta_SWA, operator.sub, deepcopy=False)
        )

    return theta_SWA, cov_diag, D


def sample_posterior_swag(theta_SWA, cov_diag, D):
    """
    Sample the inferred approximate posterior distribution found by SWAG,
    as specified by Equation (1) in their article.
    """
    # calculate total amount of weights in the torch.nn.Module
    Ns = [0]
    for tensor in theta_SWA.values():
        Ns.append(np.prod(tensor.shape))
    N = sum(Ns)

    # amount of columns in D (low-rank)
    K = len(D)

    z1 = torch.normal(torch.zeros(N), 1)
    z2 = torch.normal(torch.zeros(K), 1)

    posterior_sample = copy.deepcopy(theta_SWA)
    cumNs = np.cumsum(Ns)
    idx = 0
    for i, (key, value) in enumerate(theta_SWA.items()):
        # D is originally a list of OrderedDicts. Restructure to correctly
        # shaped torch.tensors
        layer_DT = [torch.flatten(d[key]) for d in D] # shape: (K,Ns[i])
        layer_D = torch.stack(layer_DT,axis=-1) # shape: (Ns[i],K)

        # reshape scaled samples to weight shapes
        z1_tmp = 1/np.sqrt(2) * torch.flatten(torch.sqrt(cov_diag[key])) * z1[cumNs[i]:cumNs[i+1]]
        z1_tmp = z1_tmp.reshape(posterior_sample[key].shape)

        z2_tmp = 1 / np.sqrt(2*K) * layer_D @ z2 # per "layer"
        z2_tmp = z2_tmp.reshape(posterior_sample[key].shape)

        posterior_sample[key] = posterior_sample[key] + z1_tmp + z2_tmp

    return posterior_sample

















