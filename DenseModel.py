import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseModel(nn.Module):
    def __init__(self, hidden_size):
        super(DenseModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(30, self.hidden_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.finalLinear = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        out = self.hidden(x)
        out = self.sigmoid(out)
        out = self.finalLinear(out)
        out = self.sigmoid(out)
        out = F.log_softmax(out, dim=1)
        return out
