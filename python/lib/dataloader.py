
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """This functions loads the MNIST dataset into `torch` Dataloaders.

    Parameters
    ----------
    batch_size : int
        The batch size for the training dataloader.

    Returns
    -------
    Tuple(Dataloader, Dataloader)
        The training and the test dataloader.
    """
    train_dataset = datasets.MNIST('./data',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())

    test_dataset = datasets.MNIST('./data',
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
