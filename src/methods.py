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
    def __init__(self, X, y, name):
        self.name = name # name of dataset
        self.X = X
        self.y = y[:,None] if len(y.shape) < 2 else y

        # store (per feature) normalizing parameters
        self.mux = np.mean(X,axis=0)
        self.stdx = np.std(X,axis=0)

        self.muy = np.mean(y,axis=0)
        self.stdy = np.std(y,axis=0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx, normalize=True, astorchtensor=True):
        X = (self.X[idx] - self.mux)/self.stdx if normalize else self.X[idx]
        y = (self.y[idx] - self.muy)/self.stdy if normalize else self.y[idx]
        X = torch.from_numpy(X) if astorchtensor else X
        y = torch.from_numpy(y) if astorchtensor else y
        return X, y

def torch_weight_operation(weight1, weight2, operation, deepcopy=True):
    """
    torch.nn.Module.state_dict() is an OrderedDict of weights. We want 
    to perform computations between two (or more) such objects. This is a 
    generic function to do that - element-wise.
    """
    result = copy.deepcopy(weight1) if deepcopy else weight1
    for key,value in weight1.items():
        if isinstance(weight2, (int, float)): # weight2 is a scalar
            result[key] = operation(value, weight2)
        else: # assume weight2 is an OrderedDict
            result[key] = operation(value, weight2[key])
    return result










