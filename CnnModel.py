import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnModel(nn.Module):
    def __init__(self, hidden_size):
        super(CnnModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden = nn.Conv1d(1, self.hidden_size, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.finalLinear = nn.Linear(self.hidden_size * 30, 2)

    def forward(self, x):
        out = self.hidden(x)
        out = self.sigmoid(out)
        out = out.view(-1, self.hidden_size * 30)
        out = self.finalLinear(out)
        out = self.sigmoid(out)
        out = F.log_softmax(out, dim=1)
        return out
