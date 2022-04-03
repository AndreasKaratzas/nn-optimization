
import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    """This is feed forward neural network 
    architecture defined to solve the MNIST 
    dataset.
    """
    def __init__(self):
        super(model, self).__init__()
        self.input_l = nn.Linear(784, 150)
        self.hidden_1 = nn.Linear(150, 100)
        self.hidden_2 = nn.Linear(100, 50)
        self.output_l = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.sigmoid(self.input_l(x))
        x = torch.sigmoid(self.hidden_1(x))
        x = torch.sigmoid(self.hidden_2(x))
        x = torch.sigmoid(self.output_l(x))
        return x
