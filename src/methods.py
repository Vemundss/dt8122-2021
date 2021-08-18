"""
Diverse methods used throughout the project across algorithms
"""
import numpy as np
import torch
import copy


def load_data(dataset_path):
    """
    Dataloader obeying the given dataset formats

    Args:
        dataset_path: (str) path to dataset
    """
    with open(dataset_path, "r") as f:
        X, y = [], []
        for line in f.readlines():
            units = line.split()  # split on space

            try:
                # "energy_heating_load.txt" has an extra trailing empty line
                # at the end of the file resulting in IndexError. This
                # try-except block deals with it.
                y.append(units[-1])
                X.append(units[:-1])
            except IndexError:
                pass

    train_idx = int(len(X) * 0.9)  # train/test split as specified by assignment text
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_test, y_test = X[train_idx:], y[train_idx:]

    return (
        np.array(X_train).astype(np.float32),
        np.array(y_train).astype(np.float32),
    ), (np.array(X_test).astype(np.float32), np.array(y_test).astype(np.float32))


class Dataset(torch.utils.data.Dataset):
    """
    Cast numpy tensors of input (X) and label (y) data to torch Dataset.
    torch Datasets can easily be used in torch Dataloaders which support
    built-in mini-batching, shuffling, etc.
    """

    def __init__(self, X, y, name, normalize=True, astorchtensor=True, clip_extreme_values=False):
        self.name = name  # name of dataset
        self.X = X
        self.y = y[:, None] if len(y.shape) < 2 else y

        self.normalize = normalize
        self.astorchtensor = astorchtensor
        self.clip_extreme_values = clip_extreme_values

        # store (per feature) normalizing parameters
        self.mux = np.mean(X, axis=0)
        self.stdx = np.std(X, axis=0)

        self.muy = np.mean(y, axis=0)
        self.stdy = np.std(y, axis=0)

        """
        # Tries to change normalization such that we avoid overshooting in
        # learning. Initial trials with this normalization does not appear to
        # help.
        self.minx = np.min(X, axis=0)
        self.maxx = np.max(X-self.minx, axis=0)
        self.miny = np.min(y, axis=0)
        self.maxy = np.max(y-self.miny, axis=0)
        """

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = (self.X[idx] - self.mux) / self.stdx if self.normalize else self.X[idx]
        y = (self.y[idx] - self.muy) / self.stdy if self.normalize else self.y[idx]
        
        #X = (self.X[idx] - self.minx) / self.maxx if self.normalize else self.X[idx]
        #y = (self.y[idx] - self.miny) / self.maxy if self.normalize else self.y[idx]
        
        # protein.txt has some extreme values (>50), even after a standard
        # normal standardization. This seems like corrupt data. Either way,
        # it creates such spikes in gradients that all learning ends up in weights
        # with nan-values. There are many ways to deal with this, e.g. changing
        # loss function from L2 to L1 (less sensitive to outliers) or using 
        # Adam optimizer instead of SGD. However, during sampling in SWAG, the 
        # algorithm uses SGD, thus we should also use SGD. And we should not change
        # loss function during sampling. Hence, we clip outlier values during
        # sampling of SWAG and SWAGM.
        if self.clip_extreme_values and self.normalize:
            sigmas = 3
            X = np.where(np.abs(X) > sigmas, np.sign(X)*sigmas, X)
            y = np.where(np.abs(y) > sigmas, np.sign(y)*sigmas, y)

        X = torch.from_numpy(X) if self.astorchtensor else X
        y = torch.from_numpy(y) if self.astorchtensor else y

        return X, y


def ordereddict2tensor(odict):
    """Convert odered dict of tensors to one long vector (torch.tensor)"""
    tensor = []
    for value in odict.values():
        tensor.append(torch.flatten(value))
    return torch.cat(tensor)


def tensor2ordereddict(tensor, odict):
    """Add tensor values to ordereddict"""
    n = 0
    for key, value in odict.items():
        tmp = np.prod(value.shape)
        odict[key] = tensor[n : n + tmp].reshape(value.shape)
        n += tmp
    return odict
