from typing import Tuple

import torch
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as T


def get(name: str, root: str, subset: int = None) -> Tuple[TensorDataset, TensorDataset]:
    get_data = getattr(torchvision.datasets, name)
    if name in ["MNIST", "FashionMNIST"]:
        train = get_data(root, train=True, download=True)
        t = T.Resize((32, 32))
        X_train = t(train.data.unsqueeze(1).float())
        Y_train = train.targets.reshape(-1, 1).float()
        test = get_data(root, train=False, download=True)
        X_test = t(test.data.unsqueeze(1).float())
        Y_test = test.targets.reshape(-1, 1).float()
    elif name in ["SVHN", "CIFAR10"]:
        if name == "SVHN":
            train = get_data(root, split="train", download=True)
            test = get_data(root, split="test", download=True)
            X_train, Y_train = torch.tensor(train.data).float(), torch.tensor(train.labels)
            X_test, Y_test = torch.tensor(test.data).float(), torch.tensor(test.labels)
        else:
            train = get_data(root, train=True, download=True)
            test = get_data(root, train=False, download=True)
            # original shape (N, H, W, C=3) -> (N, C=3, H, W)
            X_train = torch.tensor(train.data).permute((0, 3, 1, 2)).float()
            X_test = torch.tensor(test.data).permute((0, 3, 1, 2)).float()
            Y_train, Y_test = torch.tensor(train.targets), torch.tensor(test.targets)
        Y_train, Y_test = Y_train.reshape(-1, 1).float(), Y_test.reshape(-1, 1).float()
    else:
        raise RuntimeError(f"Dataset {name} not supported.")
    # X_train /= 255
    # X_test /= 255
    if subset:
        X_train, Y_train = X_train[:subset], Y_train[:subset]
        X_test, Y_test = X_test[:subset], Y_test[:subset]
    return TensorDataset(X_train, Y_train), TensorDataset(X_test, Y_test)
