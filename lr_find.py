import argparse
from lib.ENAS_MP import ENAS
from lib.utils import create_fastai_data
from torch.multiprocessing import get_context, cpu_count
import pickle as p

def lr_find_controller(controller, batch_size=512):
    try:
        memories = p.load(open("memories.p", "rb"))
        print(f"Successfully loaded {len(memories)} memories")
    except Exception as e:
        print("Error loading memories: ", e)
        memories = []

    return controller.controller_lr_find(controller, memories, batch_size)

def lr_find_arch(controller, batch_size=32):
    data = create_fastai_data(batch_size)
    ctx = get_context("forkserver")

    make_arch_hps = {
        "num_archs": 8
        , "num_sims": 30
    }

    with ctx.Pool() as executor:
        all_new_memories = []

        list_of_all_new_memories = list(executor.map(controller.make_architecture_mp, 
            [make_arch_hps for _ in range(cpu_count())]))

    for lst in list_of_all_new_memories:
        for sub_list in lst:
            all_new_memories.append(sub_list)

    archs = []
    for i, new_memories in enumerate(all_new_memories):
        decisions = new_memories[-1]["decisions"]

        arch = controller.create_arch_from_decisions(decisions)
        if controller.has_cuda:
            arch = arch.cuda()
        # arch_optim = optim.Adam(arch.parameters(), lr=5e-5) #5e-5
        # arch.train()
        archs.append(arch)

    return [controller.arch_lr_find(arch, data) for arch in archs]

def main(lr_find_type):
    controller = ENAS()
    if controller.has_cuda:
        controller = controller.cuda()

    if lr_find_type.lower() is "controller":
        lr_find_controller(controller)
    elif lr_find_type.lower() is "arch":
        lr_find_arch(controller)
    else:
        print("Unknown command, select <controller> or <arch>")