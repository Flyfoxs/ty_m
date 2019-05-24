import torch
from torchvision import datasets,transforms
import torchvision
from torch.autograd import  Variable
import numpy as np
import matplotlib.pyplot as plt


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

data_train=datasets.MNIST(root="./data",  transform=transform, train=True,
                          download=True
                          )
data_test=datasets.MNIST(root="./data", transform=transform, train=False)


