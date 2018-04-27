from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lib.ENAS import ENAS
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

def create_data_loaders(train_batch_size, test_batch_size):
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
        testset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

def main(max_memories=None, controller_batch_size=512, num_train_iters=60,
        train_batch_size=8, test_batch_size=64): 

    if max_memories is None:
        max_memories = controller_batch_size*3
    #batch_size=4, num_train_iters=100 is good
    #batch_size=8, num_train_iters=50 is good
    #batch_size=16, num_train_iters=30 has been working well
    try:
        memories = p.load(open("memories.p", "rb"))
        print(f"Successfully loaded {len(memories)} memories")
    except Exception as e:
        print("Error loading memories: ", e)
        memories = []

    trainloader, testloader = create_data_loaders(train_batch_size, test_batch_size)
    controller = ENAS()
    try:
        state_dict = torch.load('controller.p')
        controller.load_state_dict(state_dict)
        print("Successfully loaded controller")
    except Exception as e:
        print("Error loading controller weights: ", e)
        pass
    controller.eval()

    controller_optim = optim.SGD(params=controller.parameters(), lr=.4, momentum=.9)

    cnt = 0
    while True:
        print("Iteration {}".format(cnt))
        arch, new_memories = controller.make_architecture()
        # arch_optim = optim.SGD(arch.parameters(), lr=0.007, momentum=.9) #.01
        arch_optim = optim.Adam(arch.parameters(), lr=5e-5) #5e-5
        arch.train()

        for i, (inputs, targets) in enumerate(trainloader):
            arch_optim.zero_grad()
            outputs = arch(inputs)
            train_loss = F.nll_loss(outputs, targets)
            if i == 0:
                first_loss = train_loss.item()
            train_loss.backward()
            arch_optim.step()
            if i > num_train_iters:
                break

        final_loss = train_loss.item()

        print(f"First loss: {first_loss}, Final loss: {final_loss}")
        if final_loss > first_loss:
        # if final_loss > first_loss*.97:
            print(f"Score: 0")
            score = -1
        else:
            arch.eval()
            for inputs, targets in testloader:
                outputs = arch(inputs)
                pred = outputs.data.max(1, keepdim=True)[1]
                correct = pred.eq(targets.data.view_as(pred)).float().sum()
                score = correct/len(targets)
                print(f"Score: {score}")
                score *= 2
                score -= 1
                score = score.item()
                break

        for memory in new_memories:
            memory["score"] = score
        memories.extend(new_memories)

        if cnt % 5 == 4:
            memories = memories[-max_memories:]
            print(f"Num memories: {len(memories)}")
            p.dump(memories, open("memories.p", "wb"))
            p.dump(memories, open("memories1.p", "wb"))            
            print("Successfully saved memories")

        if cnt % 15 == 0 and len(memories) > len(memories)//10:
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

            controller.fastai_train(controller, memories, controller_batch_size)
            # normal_train(controller, controller_optim, memories[:128], controller_batch_size)
            torch.save(controller.state_dict(), 'controller1.p')
            torch.save(controller.state_dict(), 'controller.p')
            print("Successfully saved controller")    

        cnt += 1

def normal_train(controller, controller_optim, memories, batch_size, num_batches=100):
    controller.train()
    controller.memories = memories
    controller.batch_size = batch_size
    for _ in range(num_batches):
        controller_optim.zero_grad()

        loss = controller.train_controller()

        print("Loss:", loss.data.numpy())

        loss.backward()

        controller_optim.step()
    controller.memories = None
    controller.eval()

if __name__ == "__main__":
    main()
