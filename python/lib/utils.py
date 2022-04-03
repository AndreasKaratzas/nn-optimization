
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from typing import List


def colorstr(options: List[str], string_args: List[str]) -> str:
    """Usage:
    
    >>> args = ['Andreas', 'Karatzas']
    >>> print(
    ...    f"My name is {colorstr(options=['red', 'underline'], string_args=args)} "
    ...    f"and I like {colorstr(options=['bold', 'cyan'], string_args=list(['Python']))} "
    ...    f"and {colorstr(options=['cyan'], string_args=list(['C++']))}\n")
    Parameters
    ----------
    options : List[str]
        The color options.
    string_args : List[str]
        The string input for color editing.
    Returns
    -------
    str
        The colored string result.
    """
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
    colors = {'black':          '\033[30m', # basic colors
              'red':            '\033[31m',
              'green':          '\033[32m',
              'yellow':         '\033[33m',
              'blue':           '\033[34m',
              'magenta':        '\033[35m',
              'cyan':           '\033[36m',
              'white':          '\033[37m',
              'bright_black':   '\033[90m', # bright colors
              'bright_red':     '\033[91m',
              'bright_green':   '\033[92m',
              'bright_yellow':  '\033[93m',
              'bright_blue':    '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan':    '\033[96m',
              'bright_white':   '\033[97m',
              'end':            '\033[0m',  # miscellaneous
              'bold':           '\033[1m',
              'underline':      '\033[4m'}
    res = []
    for substr in string_args:
        res.append(''.join(colors[x] for x in options) +
        f'{substr}' + colors['end'])
    space_char = ''.join(colors[x] for x in options) + ' ' + colors['end']
    return space_char.join(res)


def visualize_dataset(train_loader: DataLoader, limit: int = 10):
    """This function helps in the visualization of the MNIST dataset.

    Parameters
    ----------
    train_loader : DataLoader
        This is the dataloader where we will pull data to visualize.
    limit : int, optional
        This is the number of dataset samples to visualize, by default 10
    """
    pltsize = 1

    plt.figure(figsize=((limit + 1)*pltsize, pltsize))

    for i, (X_train, y_train) in enumerate(train_loader):
        plt.subplot(1, limit + 1, i + 1)
        plt.axis('off')
        plt.imshow(X_train[:, :].numpy().reshape(28, 28), cmap="gray_r")
        plt.title('Class: '+str(y_train.item()))

        if i + 1 > limit:
            break
    
    plt.show()

def load_pretrained_model(pretrained_model_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Loads a pretrained model.

    Parameters
    ----------
    pretrained_model_path : str
        The path to the file containing the weights of the pretrained model.
    model : torch.nn.Module
        The model instance to be synchronized with the pretrained weights.

    Returns
    -------
    torch.nn.Module
        The updated model instance.
    """
    return model.load_state_dict(torch.load(pretrained_model_path))

def augment_target(y_sample: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts a class label into a one hot vector.

    Parameters
    ----------
    y_sample : torch.Tensor
        The target label.
    num_classes : int
        The total number of classes in the dataset.

    Returns
    -------
    torch.Tensor
        The augmented vector.
    """
    return F.one_hot(y_sample, num_classes=num_classes).to(torch.float32)
