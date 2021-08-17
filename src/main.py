"""
Main file as specified by assignment (ProbAI2021).

- As an alternative, the project contains jupyter notebooks for each algorithm
implemented. This is, in my opinion, a superior way to inspect the implementations.
"""
import argparse
import sys
import os
import shutil

import numpy as np
import torch

# pyro imports
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

# custom imports
from methods import *
from nnmodules import GenericDNN
from swag import *
from swagm import *
from evaluation import *
from mnd import *


def get_parameters():
    """Handle user inputs: dataset and method choice"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="yacht.txt", help="Which dataset to use", type=str
    )
    parser.add_argument("--method", default=0, help="Which method to use", type=int)
    args = parser.parse_args()

    dataset_name = args.dataset.split("/")[-1]  # dataset name - without path specs
    dataset_path = "../datasets/" + dataset_name

    try:
        f = open(dataset_path, "rb")
        f.close()
    except OSError:
        print(f"Could not open/read file with path: {dataset_path!r}. Exiting..")
        sys.exit()

    # Check if chosen method is contained in implemented methods wrt.
    # assignemnt numbering
    if args.method not in [0, 1, 3]:
        raise NotImplementedError

    return dataset_path, args.method


def get_dataloaders(dataset_path):
    """
    Helper-function for initialising trainloader and testloader
    (Same for every method)
    """
    dataset_name = dataset_path.split("/")[-1]
    # load data and create torch training data loader
    (X_train, y_train), (X_test, y_test) = load_data(dataset_path)
    trainloader = torch.utils.data.DataLoader(
        Dataset(X_train, y_train, dataset_name.split(".")[0]),
        batch_size := 32,
        shuffle := True,
    )

    # init test loader, and set trainloader normalizing quantities
    # to be used by the testloader as well.
    testloader = torch.utils.data.DataLoader(
        Dataset(X_test, y_test, dataset_name.split(".")[0]), shuffle=False
    )
    testloader.dataset.mux = trainloader.dataset.mux
    testloader.dataset.stdx = trainloader.dataset.stdx
    testloader.dataset.muy = trainloader.dataset.muy
    testloader.dataset.stdy = trainloader.dataset.stdy
    return trainloader, testloader


def swag(dataset_path):
    """
    Train and test a SWAG-model

    Here, one is free to alter hyper-parameters if wanted
    """
    trainloader, testloader = get_dataloaders(dataset_path)

    # init model, criterion and optimizer
    net = GenericDNN(
        input_size := len(trainloader.dataset[0][0]),
        hidden_size := input_size,
        output_size := 1,
    )
    criterion = torch.nn.MSELoss()  # L1Loss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train and save weights
    checkpoint_path = "../checkpoints/"
    if train := True:
        # delete previous checkpoints for model with current dataset
        if delete_previous_checkoints := True:
            try:
                shutil.rmtree(checkpoint_path + trainloader.dataset.name)
            except FileNotFoundError:
                # can't delete non-existing file... just carry on
                pass

        # train and save weights
        train_swag(
            net,
            trainloader,
            optimizer,
            criterion,
            init_epochs=1000,
            sampling_epochs=5,
            nsamples=200,
            path_to_checkpoints=checkpoint_path,
        )

    # Do inference (infer approximate sample mean and covariance of an assumed Guassian posterior)
    theta_SWA, cov_diag, D = inference_swag(trainloader.dataset.name, checkpoint_path)

    # Do predictions
    Lpi, Upi, mu_swa, mu_pred = monte_carlo_PI(
        testloader.dataset[:][0],
        net,
        theta_SWA,
        cov_diag,
        D,
        nsamples=50,
        percentile=0.9,
    )

    # invert prediction normalization
    invert_normalization = (
        lambda y: y * trainloader.dataset.stdy + trainloader.dataset.muy
    )
    Lpi, Upi, mu_swa, mu_pred, y_true = map(
        invert_normalization, (Lpi, Upi, mu_swa, mu_pred, testloader.dataset[:][1])
    )

    # evaluate model
    print("--- EVALUATE MODEL WITH ASSIGNMENT SPECIFIED METRICS ---")
    # mu_pred or mu_swa?
    print(f"{rmse(y_true, mu_pred)=}, {picp(Lpi, Upi, y_true)=}, {mpiw(Lpi, Upi)=}")


def mnd(dataset_path):
    """
    Train and test a matrix normal distribution model
    """
    trainloader, testloader = get_dataloaders(dataset_path)

    net = MND_BNN_alternative(
        input_size := len(trainloader.dataset[0][0]),
        hidden_size := input_size,
        output_size := 1,
    )

    def train(svi, train_loader, batch_size, use_cuda=False):
        epoch_loss = 0.0

        for i, (X, y) in enumerate(train_loader):
            epoch_loss += svi.step(X, y)

        N = len(train_loader.dataset)
        return epoch_loss / N

    # clear param store
    # pyro.clear_param_store()

    # setup the optimizer
    optimizer = Adam({"lr": 1.0e-3})
    svi = SVI(net.model, net.guide, optimizer, loss=Trace_ELBO())
    # svi = SVI(net.model, AutoMultivariateNormal(net.model), optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS := 5000):
        total_epoch_loss_train = train(svi, trainloader, trainloader.batch_size)
        train_elbo.append(-total_epoch_loss_train)
        if not (epoch % 100):
            print(
                "[epoch %03d]  average training loss: %.4f"
                % (epoch, total_epoch_loss_train)
            )

    # do predictions
    Lpi, Upi, mu_posterior, mu_pred = net.monte_carlo_PI(
        testloader.dataset[:][0], nsamples=50, percentile=0.95
    )

    # invert prediction normalization
    invert_normalization = (
        lambda y: y * trainloader.dataset.stdy + trainloader.dataset.muy
    )
    Lpi, Upi, mu_posterior, mu_pred, y_true = map(
        invert_normalization,
        (Lpi, Upi, mu_posterior, mu_pred, testloader.dataset[:][1]),
    )

    # evaluate model
    print("--- EVALUATE MODEL WITH ASSIGNMENT SPECIFIED METRICS ---")
    # mu_pred or mu_posteriormu_posterior?
    print(f"{rmse(y_true, mu_pred)=}, {picp(Lpi, Upi, y_true)=}, {mpiw(Lpi, Upi)=}")


def swagm(dataset_path):
    """
    Custom (novel) extension to the SWAG algorithm. In short, posterior
    is now a mixture of gaussians rather than a gaussian. Additionally, new
    mixtures are only added if they are not already described by an exisiting
    mixture.
    """
    trainloader, testloader = get_dataloaders(dataset_path)

    # init model, criterion and optimizer
    def init_net():
        """
        Easily reinstantiate net (reinitialise weights), criterion and
        optimizer with same hyper-params
        """
        net = GenericDNN(
            input_size := len(trainloader.dataset[0][0]),
            hidden_size := input_size,
            output_size := 1,
        )
        criterion = torch.nn.MSELoss()  # L1Loss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        return net, criterion, optimizer

    # Set 'train:=True' if model not previously trained, or want to train a new model
    checkpoint_path = "../checkpoints/"
    if train := True:

        # delete previous checkpoints for model with current dataset
        if delete_previous_checkoints := True:
            try:
                shutil.rmtree(checkpoint_path + trainloader.dataset.name + "-swagm")
            except FileNotFoundError:
                # can't delete non-existing file... just carry on
                pass

        # train and save weights
        train_swagm(
            init_net=init_net,
            nmixtures=5,
            trainloader=trainloader,
            init_epochs=1000,
            sampling_epochs=5,
            nsamples=200,
            path_to_checkpoints=checkpoint_path,
        )

    # novelty - new mixture should only be a new mixture if it is not equal
    # to other mixtures.
    collect_similar_modes(
        name=trainloader.dataset.name,
        nsamples=1000,
        alpha=0.05,
        checkpoint_path="../checkpoints/",
    )

    # Do predictions
    net, criterion, optimizer = init_net()
    theta_SWAs, cov_diags, Ds, catdist = inference_swagm(
        trainloader.dataset.name, checkpoint_path
    )
    Lpi, Upi, mu_pred = monte_carlo_PI_swagm(
        testloader.dataset[:][0],
        net,
        theta_SWAs,
        cov_diags,
        Ds,
        catdist,
        nsamples=50,
        percentile=0.9,
    )

    # invert prediction normalization
    invert_normalization = (
        lambda y: y * trainloader.dataset.stdy + trainloader.dataset.muy
    )
    Lpi, Upi, mu_pred, y_true = map(
        invert_normalization, (Lpi, Upi, mu_pred, testloader.dataset[:][1])
    )

    # evaluate model
    print("--- EVALUATE MODEL WITH ASSIGNMENT SPECIFIED METRICS ---")
    print(f"{rmse(y_true, mu_pred)=}, {picp(Lpi, Upi, y_true)=}, {mpiw(Lpi, Upi)=}")


"""Simple check for get_parameters()"""
dsp, method_num = get_parameters()
print(f"Echo arguments: {dsp=} {method_num=}")

if method_num == 0:
    swag(dsp)
elif method_num == 1:
    mnd(dsp)
    pass
elif method_num == 3:
    swagm(dsp)
