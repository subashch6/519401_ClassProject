import pandas as pd
import numpy as np
import random
import torch


data = pd.read_csv("data.csv", index_col='id')
data = data.drop(columns=['Unnamed: 32'])

print(data)
print(data.shape)