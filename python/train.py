
import warnings
import argparse

import torch

from lib.dataloader import load_mnist
from lib.engine import train, test
from lib.model import model
from lib.utils import colorstr, visualize_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='PyTorch Feed Forward Neural Network trained on MNIST dataset.')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='Number of classes in dataset.')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size (default: 1).')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='Number of total epochs to run (default: 10).')
    parser.add_argument('--lr', default=1e-1, type=float,
                        help='Initial learning rate (default: 1e-1).')
    parser.add_argument('--visual', default=False, action='store_true',
                        help='Visualize some dataset samples (default: False).')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU option.')
    args = parser.parse_args()

    # disable user warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(
        f"Device utilized: {colorstr(options=['red', 'underline'], string_args=list([device]))}.\n")

    # load dataset
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)

    # initialize model
    net = model().to(device)

    # initialize optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    # initialize error function
    criterion = torch.nn.MSELoss()

    # log model architecture
    print(net)

    # visualize some dataset samples
    if args.visual:
        visualize_dataset(train_loader=train_loader)

    # fit model
    for epoch in range(args.epochs):
        train(
            model=net, 
            train_loader=train_loader, 
            device=device, 
            optimizer=optimizer, 
            criterion=criterion, 
            epoch=epoch,
            num_classes=args.num_classes
        )
    
    # evaluate model
    test(
        model=net,
        test_loader=test_loader,
        device=device,
        num_classes=args.num_classes
    )

    # save model
    torch.save(net.state_dict(), './data/net.pt')
