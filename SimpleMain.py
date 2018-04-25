from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lib import FLAGS
from lib.SimpleENAS import SimpleENAS
from ipdb import set_trace

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import torchvision
import torch
import torch.optim as optim
from torch.autograd import Variable

from copy import deepcopy

from tqdm import tqdm

import numpy as np

import pickle as p

from random import shuffle

def create_data_loaders(batch_size):
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
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

def main(batch_size=64, max_memories=3000):
    try:
        memories = p.load(open("memories.p", "rb"))
        print(f"Successfully loaded {len(memories)} memories")
    except Exception as e:
        print("Error loading memories: ", e)
        memories = []

    trainloader, testloader = create_data_loaders(batch_size)
    controller = SimpleENAS()
    try:
        state_dict = torch.load('controller.p')
        controller.load_state_dict(state_dict)
        print("Successfully loaded controller")
    except Exception as e:
        print("Error loading controller weights: ", e)
        pass
    controller.eval()

    cnt = 0
    while True:
        print("Iteration {}".format(cnt))
        arch, new_memories = controller.make_architecture()
        arch_optim = optim.SGD(arch.parameters(), lr=0.7, momentum=.9) #.08
        arch.train()

        for inputs, targets in trainloader:
            arch_optim.zero_grad()
            outputs = arch(inputs)
            train_loss = F.nll_loss(outputs, targets)
            train_loss.backward()
            arch_optim.step()
            break

        arch.eval()
        for inputs, targets in testloader:
            outputs = arch(inputs)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct = pred.eq(targets.data.view_as(pred)).float().sum()
            score = correct/len(targets)
            print(f"Score: {score}")
            score *= 2
            score -= 1
            for memory in new_memories:
                memory["score"] = score
            memories.extend(new_memories)
            break

        if cnt % 30 == 0:
            print(len(memories))            
            memories = memories[-max_memories:]
            print(len(memories))
            p.dump(memories, open("memories.p", "wb"))
            p.dump(memories, open("memories1.p", "wb"))            
            print("Successfully saved memories")

        if cnt % 30 == 0 and len(memories) > max_memories/2:
            # scores = []
            # for memory in memories:
            #     scores.append(memory["score"])

            # scores = np.array(scores)
            # scores -= scores.mean()
            # scores /= scores.std()
            # scores = (scores - scores.min()) / (scores.max() - scores.min())
            # scores = Variable(torch.from_numpy(scores).float())

            # temp_memories = deepcopy(memories)
            # for memory, score in zip(temp_memories, scores):
            #     memory["score"] = score

            controller.fastai_train(controller, memories, batch_size)
            torch.save(controller.state_dict(), 'controller1.p')
            torch.save(controller.state_dict(), 'controller.p')
            print("Successfully saved controller")    

        cnt += 1

if __name__ == "__main__":
    main()
