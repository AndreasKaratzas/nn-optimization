
import time
import torch
import torch.nn.functional as F

from lib.utils import augment_target

def train(
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        device: torch.device, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        epoch: int,
        num_classes: int
    ):
    """This function runs one training epoch on a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model for training.
    train_loader : torch.utils.data.DataLoader
        The training data loader.
    device : torch.device
        The chosen device. Might be an NVIDIA GPU if available.
    optimizer : torch.optim.Optimizer
        The optimizer declared for the training of the model.
    criterion : torch.nn.Module
        The loss function declared for the computation of the prediction error.
    epoch : int
        The epoch counter.
    num_classes : int
        The number of classes for the MNIST dataset.
    """

    # Set model to training mode
    model.train()

    # Initialize a `top1` accuracy metric
    correct = 0

    # Start epoch benchmark
    start_time = time.time()

    # Loop over each batch from the training set
    for data, target in train_loader:
        # Copy data to GPU if needed
        # NOTE: Commented out to optimize latency
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, augment_target(
            y_sample=target, num_classes=num_classes))

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        # Get top1 prediction
        pred = output.data.max(1, keepdim=True)[1]

        # Update the number of total accurate predictions
        correct += pred.eq(target.data).cpu().sum()
    
    # End epoch benchmark
    end_time = time.time()

    print(
        f'[EPOCH {epoch:4d}] [LOSS {loss.data.item():6.5f}] '
        f'[ACCURACY {correct.item():6d} out of {60000:6d}] '
        f'Work took {(end_time - start_time):6.1f} seconds'
    )


def test(
        model: torch.nn.Module, 
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_classes: int
    ):
    """This function tests a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model for evaluation.
    test_loader : torch.utils.data.DataLoader
        The evaluation dataset.
    device : torch.device
        A pointer to a GPU device if available.
    num_classes : int
        The number of classes for the MNIST dataset.
    """

    # Initialize a `top1` accuracy metric
    correct = 0

    # Initialize a `loss` metric
    test_loss = 0

    # Set model to evaluation mode
    model.eval()

    # Start epoch benchmark
    start_time = time.time()

    # Deactivate Grad computation
    with torch.no_grad():

        # Loop over each batch from the validation set
        for data, target in test_loader:
            # Copy data to GPU if needed
            # NOTE: Commented out to optimize latency
            data = data.to(device)
            target = target.to(device)

            # Pass data through the network
            output = model(data)

            # Calculate loss
            test_loss += F.mse_loss(output, augment_target(y_sample=target, 
            num_classes=num_classes), size_average=False).item()

            # Get top1 prediction
            pred = output.data.max(1, keepdim=True)[1]

            # Update the number of total accurate predictions
            correct += pred.eq(target.data).cpu().sum()
    
    # End epoch benchmark
    end_time = time.time()

    # Average the calculated evaluation model loss
    test_loss /= len(test_loader.dataset)

    print(
        f'[EVALUATION] [LOSS {test_loss:6.5f}] '
        f'[ACCURACY {correct.item():6d} out of {60000:6d}] '
        f'Work took {(end_time - start_time):6.1f} seconds'
    )
