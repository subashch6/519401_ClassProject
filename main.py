import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils import data as torchdata
import random
from DenseModel import DenseModel
from CnnModel import CnnModel


model_type = 'dense'

model = DenseModel(100)
if model_type != 'dense':
    model = CnnModel(100)

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)

optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)

print("device: {}".format(device))



data = pd.read_csv("data.csv", index_col='id')
data = data.drop(columns=['Unnamed: 32'])

print(data.shape)

train_data_length = int(data.shape[0] * .8)
test_data_length = data.shape[0] - train_data_length

x = data.drop(columns=['diagnosis']).values.astype(np.float32)
eye = torch.eye(2)
y = data['diagnosis'].values
y = np.where(y == 'M', 1, 0).astype(np.long)

sample = random.sample(range(data.shape[0]), train_data_length)
not_sample = [i for i in range(data.shape[0]) if i not in sample]

if model_type != 'dense':
    x = x[:, np.newaxis, :]

trainx = x[sample]
trainy = y[sample]
testx = x[not_sample]
testy = y[not_sample]

print(trainx.shape)
print(trainy.shape)
print(testx.shape)
print(testy.shape)

trainx = torch.from_numpy(trainx)
trainy = torch.from_numpy(trainy).type(torch.long)
testx = torch.from_numpy(testx)
testy = torch.from_numpy(testy).type(torch.long)

train_dataset = torchdata.TensorDataset(trainx, trainy)
test_dataset = torchdata.TensorDataset(testx, testy)


batch_size = 500
train_dataloader = torchdata.DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = torchdata.DataLoader(test_dataset, batch_size=batch_size)




def train(epoch):
    total_loss = 0
    num_loops = 0
    model.train()
    for batch_idx, (x,y) in enumerate(train_dataloader):
        num_loops += 1
        #print("x: {}, y: {}".format(x, y))
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print("epoch number: {}, batch index: {}, loss: {},".format(epoch, batch_idx, total_loss/num_loops))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    dataset_length = len(test_dataloader.dataset) * 1.0
    print("epoch number: {}, test loss: {}, test accuracy: {}".format(epoch, test_loss, correct/dataset_length))


if __name__ == "__main__":
    for epoch in range(1, 501):
        train(epoch)
        if epoch % 50 == 0:
            test(epoch)
