from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import torch
from tqdm import tqdm
import numpy as np

from fastai.imports import *
from fastai.transforms import *
from fastai.learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

def create_data_loaders(train_batch_size, test_batch_size, cuda):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    path = "./data"

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader

def create_fastai_data(batch_size):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    path = "./data"

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    trn_X = []
    trn_y = []    
    for i, (x, y) in enumerate(tqdm(trainloader)):
        trn_X.append(x.numpy())
        trn_y.append(y.numpy())
        if i > len(trainloader)//30:
            break

    val_X = []
    val_y = [] 
    for i, (x, y) in enumerate(testloader):
        val_X.append(x.numpy())
        val_y.append(y.numpy())
        if i > len(testloader)//30:
            break

    trn_X = np.concatenate(trn_X)
    trn_y = np.concatenate(trn_y)

    val_X = np.concatenate(val_X)
    val_y = np.concatenate(val_y)
    

    trn = [trn_X, trn_y]
    val = [val_X, val_y]

    data = ImageClassifierData.from_arrays(path, trn=trn, val=val,
                                        classes=classes)

    return data