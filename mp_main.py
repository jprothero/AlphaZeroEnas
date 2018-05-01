from lib.ENAS_MP import ENAS
from ipdb import set_trace

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
from lib.utils import create_data_loaders

from torch.multiprocessing import Pool, get_context, cpu_count

from copy import deepcopy

from tqdm import tqdm

import numpy as np

import pickle as p

from random import shuffle
from concurrent.futures import ProcessPoolExecutor as PPExec
from concurrent.futures import ThreadPoolExecutor as TPE

def main(args, max_memories=100000, num_train_iters=25,
        train_batch_size=32, test_batch_size=64): 

    num_sims = int(args.num_sims)
    num_archs = int(args.num_archs)
    num_concurrent = int(args.num_concurrent)
    controller_batch_size = int(args.controller_batch_size)
    min_memories = int(args.min_memories) if args.min_memories is not None else None
    if min_memories is None:
        min_memories = max_memories // 100

    #batch_size=4, num_train_iters=100 is good
    #batch_size=8, num_train_iters=50 is good
    #batch_size=16, num_train_iters=30 has been working well
    try:
        memories = p.load(open("memories.p", "rb"))
        print(f"Successfully loaded {len(memories)} memories")
    except Exception as e:
        print("Error loading memories: ", e)
        memories = []

    controller = ENAS()
    trainloader, testloader = create_data_loaders(train_batch_size, test_batch_size, cuda=controller.has_cuda)
    if controller.has_cuda:
        controller = controller.cuda()

    try:
        state_dict = torch.load('controller.p')
        controller.load_state_dict(state_dict)
        print("Successfully loaded controller")
    except Exception as e:
        print("Error loading controller weights: ", e)
        pass

    # controller_optim = optim.SGD(params=controller.parameters(), lr=.4, momentum=.9)

    ctx = get_context("forkserver")
    cnt = 0

    try:
        max_score = p.load(open("max_score.p", "rb"))
        max_score_decisions = p.load(open("max_score_decisions.p", "rb"))
    except:
        max_score = -1
        max_score_decisions = None

    while True:    
        make_arch_hps = {
            "num_archs": num_archs
            , "num_sims": num_sims
        }

        print("Iteration {}".format(cnt))
        controller.eval()

        if num_concurrent > 1:
            all_new_memories = []

            with ctx.Pool() as executor:
                list_of_all_new_memories = list(executor.map(controller.make_architecture_mp, 
                    [make_arch_hps for _ in range(num_concurrent)]))

            # with TPE(macro_max_workers) as executor:
            #     list_of_all_new_memories = list(executor.map(controller.make_architecture_mp, 
            #         [make_arch_hps for _ in range(num_concurrent)]))

            #so the above return [[memories, memories], [memories, memories]]
            #and what we want is [memories, memories, memories, memories]
            for lst in list_of_all_new_memories:
                for sub_list in lst:
                    all_new_memories.append(sub_list)
        else:
            all_new_memories = controller.make_architecture_mp(make_arch_hps)

        for i, new_memories in enumerate(all_new_memories):
            decisions = new_memories[-1]["decisions"]

            arch = controller.create_arch_from_decisions(decisions)
            if controller.has_cuda:
                arch = arch.cuda()
            arch_optim = optim.Adam(arch.parameters(), lr=5e-3) #5e-5 was good
            arch.train()

            for i, (inputs, targets) in enumerate(trainloader):
                if controller.has_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
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
                print(f"Score: 0")
                score = -1
            else:
                arch.eval()
                for inputs, targets in testloader:
                    if controller.has_cuda:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    outputs = arch(inputs)
                    pred = outputs.data.max(1, keepdim=True)[1]
                    correct = pred.eq(targets.data.view_as(pred)).float().sum()
                    score = correct/len(targets)
                    if score > max_score:
                        max_score = score
                        max_score_decisions = decisions
                    print(f"Score: {score}")
                    score *= 2
                    score -= 1
                    score = score.item()
                    break

            for memory in new_memories:
                memory["score"] = score

            memories.extend(new_memories) 
            cnt += 1

        memories = memories[-max_memories:]
        print(f"Num memories: {len(memories)}")
        p.dump(memories, open("memories.p", "wb"))
        p.dump(memories, open("memories_backup.p", "wb"))            
        print("Successfully saved memories")

        if max_score_decisions is not None:
            p.dump(max_score, open("max_score.p", "wb"))
            p.dump(max_score, open("max_score_backup.p", "wb"))  

            p.dump(max_score_decisions, open("max_score_decisions.p", "wb"))
            p.dump(max_score_decisions, open("max_score_decisions_backup.p", "wb"))  

        controller.fastai_train(controller, memories, controller_batch_size, min_memories=max_memories//100)
        # normal_train(controller, controller_optim, memories[:128], controller_batch_size)
        torch.save(controller.state_dict(), 'controller.p')
        torch.save(controller.state_dict(), 'controller_backup.p')
        print("Successfully saved controller")   

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sims", default=3)
    parser.add_argument("--num_archs", default=1)
    parser.add_argument("--num_concurrent", default=cpu_count())
    parser.add_argument("--min_memories", default=None)
    parser.add_argument("--controller_batch_size", default=512)
    args = parser.parse_args()

    main(args)
