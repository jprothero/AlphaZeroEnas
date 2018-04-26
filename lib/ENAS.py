import torch.nn as nn
from .qrnn import QRNN
from torch.nn import functional as F
from torch.distributions import Categorical
import torch
import numpy as np
from torch.autograd import Variable
from ipdb import set_trace
from .AlphaZero import AlphaZero
from random import sample

from .fastai.imports import *
from .fastai.transforms import *
from .fastai.learner import *
from .fastai.model import *
from .fastai.dataset import *
from .fastai.sgdr import *
from .fastai.plots import *

from copy import deepcopy as c

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class FastaiWrapper():
    def __init__(self, model, crit):
        self.model = model
        self.crit = crit
    
    def get_layer_groups(self, precompute=False):
        return self.model

class ENAS(nn.Module):
    def __init__(self, num_classes=10, R=32, C=32, CH=3, num_layers=4, lstm_size=70, 
            num_lstm_layers=4, value_head_dims=64):
        super(ENAS, self).__init__()
        self.num_classes = num_classes
        self.R = R
        self.C = C
        self.CH = CH
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.num_lstm_layers = num_lstm_layers
        
        self.validating = False
        
        self.controller = QRNN(lstm_size, lstm_size,
                               num_layers=num_lstm_layers)

        self.fake_data = self.create_fake_data()

        self.value_head = nn.Sequential(*[
            nn.Linear(lstm_size, value_head_dims)
            , LayerNorm(value_head_dims)
            , nn.ReLU()
            , nn.Linear(value_head_dims, value_head_dims)
            , LayerNorm(value_head_dims)      
            , nn.ReLU()
            , nn.Linear(value_head_dims, 1)
            , nn.Tanh()
        ])

        self.filters = [
            16,
            32,
            64,
            # 128,
        ]

        self.groups = []

        i = 0
        while True:
            self.groups.append(2**i)
            if 2**i == self.filters[-1]:
                break
            i += 1

        self.kernels = [
            1,
            3,
        ]

        self.dilations = [
            1,
            2,
            # 4,
        ]

        self.activations = [
            # nn.Softmax(dim=1),
            nn.Tanh(),
            # nn.Sigmoid(),
            nn.ReLU(),
            # lambda x: x
        ]

        self.strides = [
            1,
            2,
            # 3
        ]

        self.skips = [i for i in range(num_layers-1)]

        self.decisions = {
            "filters": self.filters,
            "groups": self.groups,
            "kernels": self.kernels,
            "dilations": self.dilations,
            "activations": self.activations,
            "strides": self.strides,
            "skips": self.skips
        }

        total_embeddings = 0
        for _, lst in self.decisions.items():
            total_embeddings += len(lst)

        self.decision_list = [
            "filters"
            , "groups"
            , "skips"
            , "kernels"
            , "dilations"
            , "activations"
            , "strides"
        ]

        self.total_embeddings = total_embeddings + num_layers

        self.emb_merge_pre_softmax = nn.Sequential(*[
                LayerNorm(lstm_size),
                nn.Linear(lstm_size, self.total_embeddings), 
            ])

        def skip_condition(layer_idx):
            if layer_idx > 1:
                return False
            else:
                return True

        def filter_condition(layer_idx):
            if layer_idx == 0:
                return False
            else:
                return True

        self.decision_conditions = dict()

        for name in self.decision_list:
            if name is "skips":
                self.decision_conditions[name] = skip_condition
            elif name is "filters":
                self.decision_conditions[name] = filter_condition
            else:
                self.decision_conditions[name] = lambda x: False

        self.mask_conditions = dict()

        def skip_mask(layer_idx, probas):
            probas[layer_idx-1:] = 0
            return probas

        for name in self.decision_list:
            if name is "skips":
                self.mask_conditions[name] = skip_mask
            else:
                self.mask_conditions[name] = None

        self.starting_indices = []

        self.first_emb = nn.Parameter(torch.rand(lstm_size)-.5).view(1, 1, -1)
        softmaxs = []
        embeddings = []
        starting_idx = 0
        for name in self.decision_list:
            options = self.decisions[name]
            softmaxs.append(nn.Sequential(*[
                LayerNorm(lstm_size),
                nn.Linear(lstm_size, len(options)), 
                nn.Softmax(dim=1)
            ]))

            embeddings.extend([nn.Parameter(torch.rand(lstm_size)-.5) for
                _ in range(len(options))])

            self.starting_indices.append(starting_idx)
            starting_idx += len(options)

        embeddings.extend([nn.Parameter(torch.rand(lstm_size)-.5) for
                _ in range(num_layers)])

        self.softmaxs = nn.ModuleList(softmaxs)
        self.embeddings = nn.ParameterList(embeddings)

    def check_condition(self, az, layer_idx, decision_name):
        condition = self.decision_conditions[decision_name]
        return condition(layer_idx)

    def cont_out_from_trajectory(self, trajectory, decision_name):
        indices = []
        weights = dict()
        seen = dict()

        scaled_one = 1/len(trajectory)
        for emb_idx in trajectory:
            try:
                seen[emb_idx]
            except:
                seen[emb_idx] = True

                try:
                    weights[emb_idx] += scaled_one
                except:
                    weights[emb_idx] = scaled_one

                indices.append(emb_idx)
        
        logits = []
        embeddings = []
        for _, idx in enumerate(indices):
            embeddings.append(self.embeddings[idx].unsqueeze(0))
            logits.append(self.emb_merge_pre_softmax(self.embeddings[idx])[indices].unsqueeze(0))

        logits = torch.cat(logits)
        logits = logits.sum(1)
        probas = F.softmax(logits)

        for proba, idx in zip(probas, indices):
            proba = proba*weights[idx]

        probas /= probas.sum() #normalize
        probas = probas.unsqueeze(-1)
        embeddings = torch.cat(embeddings)
        final_emb = embeddings * probas
        final_emb = final_emb.sum(0)
        final_emb = final_emb.view(1, 1, -1)

        cont_out = self.controller(final_emb)[0].squeeze(0)

        return cont_out

    def do_sim(self, az, cont_out):
        if az.curr_node["d"] >= az.max_depth-1:
            return

        trajectory = az.select(self.starting_indices, self.decision_list)

        #so we get to a node and we see if we should skip expanding it
        #this is the same issue as before
        #if we get to checking if we expand something we shouldnt have expanded,
        #we already f'd up
        #so I need to check future nodes

        depth = az.curr_node["d"]
        layer_idx = depth // len(self.decision_list)
        decision_idx = depth % len(self.decision_list)
        decision_name = self.decision_list[decision_idx]

        next_depth = az.curr_node["d"] + 1
        next_layer_idx = next_depth // len(self.decision_list)
        next_decision_idx = next_depth % len(self.decision_list)
        next_decision_name = self.decision_list[next_decision_idx]

        d_plus = 1
        while True:
            skip_curr = self.check_condition(az, next_layer_idx, next_decision_name)
            if not skip_curr:
                break
            else:
                depth = next_depth
                layer_idx = next_layer_idx
                decision_idx = next_decision_idx
                decision_name = next_decision_name

                next_depth += 1
                next_layer_idx = next_depth // len(self.decision_list)
                next_decision_idx = next_depth % len(self.decision_list)
                next_decision_name = self.decision_list[next_decision_idx]
                d_plus += 1

        if len(trajectory) > 0:
            cont_out = self.cont_out_from_trajectory(trajectory, decision_name)

        probas = self.softmaxs[decision_idx](cont_out).squeeze()

        probas_np = probas.detach().data.numpy()

        if self.mask_conditions[decision_name] is not None:
            probas_np = self.mask_conditions[decision_name](layer_idx, probas_np)

        az.expand(probas_np, d_plus)

        value = self.value_head(cont_out.squeeze())
        value = value.detach().data.numpy()
        az.backup(value)

    def make_architecture(self, num_sims=40):
        self.eval()
        self.filter_chosen = False
        new_memories = []
        az = AlphaZero(max_depth=self.num_layers*len(self.decision_list))

        cont_out = self.controller(self.first_emb)[0].squeeze(0)

        orig_cont_out = cont_out.clone()

        trajectory = []

        decisions = dict()
        for name in self.decision_list:
            decisions[name] = []

        while True:
            for _ in range(num_sims):
                self.do_sim(az=az, cont_out=cont_out)
                while az.curr_node["parent"] is not None:
                    az.curr_node = az.curr_node["parent"]
                    cont_out = orig_cont_out.clone()

            while az.curr_node["parent"] is not None:
                az.curr_node = az.curr_node["parent"]

            d = az.curr_node["d"]
            decision_idx = d % len(self.decision_list)
            starting_idx = self.starting_indices[decision_idx]
            name = self.decision_list[decision_idx]
            choice_idx, visits = az.select_real() 

            new_memories.append({
                "search_probas": torch.from_numpy(visits).float()
                , "trajectory": c(trajectory)
            })

            emb_idx = starting_idx + choice_idx
            
            trajectory.append(emb_idx)

            decisions[name].append(choice_idx)

            orig_cont_out = self.cont_out_from_trajectory(trajectory, name)
            
            cont_out = orig_cont_out.clone()

            if d >= az.max_depth-1:
                break

        for memory in new_memories:
            memory["decisions"] = decisions

        return self.create_arch_from_decisions(decisions), new_memories

    def create_fake_data(self):
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
        trn_X = np.zeros(shape=(64, self.CH, self.R, self.C))
        trn_y = np.zeros(shape=(64, 1))
        val_X = np.zeros(shape=(64, self.CH, self.R, self.C))
        val_y = np.zeros(shape=(64, 1))
        trn = [trn_X, trn_y]
        val = [val_X, val_y]
        fake_data = ImageClassifierData.from_arrays("./data", trn=trn, val=val,
                                        classes=classes)
        return fake_data

    def forward(self, X):
        pass

    def train_controller(self, _, __):
        batch = sample(self.memories, self.batch_size)

        search_probas = []
        policies = []

        value_loss = 0
        search_probas_loss = 0
        for memory in batch:
            choice_indices = memory["choice_indices"]
            decision_indices = memory["decision_indices"]
            score = memory["score"]
            search_probas = memory["search_probas"]

            cont_out = self.controller(self.first_emb)[0].squeeze(0)

            for choice_idx, decision_idx in zip(choice_indices, decision_indices):
                starting_idx = self.starting_indices[decision_idx]
                emb = self.embeddings[starting_idx + choice_idx].view(1, 1, -1)
                cont_out = self.controller(emb)[0].squeeze(0)
                value_loss += F.mse_loss(self.value_head(cont_out.squeeze()), score)
            probas = self.softmaxs[decision_idx](cont_out).squeeze()
            policies.append(probas)
            search_probas.append(search_probas)
        search_probas = torch.cat(search_probas)
        search_probas = search_probas.unsqueeze(0)
        search_probas = Variable(search_probas)

        policies = torch.cat(policies)
        policies = policies.unsqueeze(-1)

        dist_matching_loss = -(search_probas**2).mm(torch.log(1 - search_probas - policies))
        dist_matching_loss /= self.batch_size

        # dist_matching_loss /= len(search_probas) #might be wrong

        value_loss /= self.batch_size
        search_probas_loss /= self.batch_size
        value_loss *= 6
        total_loss = search_probas_loss + value_loss

        return total_loss

    def fastai_train(self, controller, memories, batch_size, num_cycles=12, epochs=5):
        self.training = True
        self.memories = memories
        self.batch_size = batch_size
        if (len(memories) < batch_size):
            print("Have {} memories, need {}".format(len(memories), batch_size))
            return
        controller_wrapped = FastaiWrapper(model=controller, crit=self.train_controller)
        controller_learner = Learner(data=self.fake_data, models=controller_wrapped)
        controller_learner.crit = controller_wrapped.crit
        controller_learner.model.train()

        controller_learner.model.real_forward = controller_learner.model.forward

        controller_learner.model.forward = lambda x: x
        controller_learner.fit(0.2, epochs, cycle_len=10, use_clr_beta=(10, 13.68, 0.95, 0.85), 
            wds=1e-6)

        controller_learner.model.forward = controller_learner.model.real_forward

    def LR_find(self, controller, memories, batch_size, start_lr=1e-5, end_lr=10):
        self.training = False
        self.memories = memories
        self.batch_size = batch_size
        if (len(memories) < batch_size):
            print("Have {} memories, need {}".format(len(memories), batch_size))
            return
        arch = FastaiWrapper(model=controller, crit=self.train_controller)

        arch.model.real_forward = arch.model.forward
        arch.model.forward = lambda x: x

        learn = Learner(data=self.fake_data, models=arch)
        learn.crit = arch.crit
        learn.model.train()

        learn.lr_find(start_lr=start_lr, end_lr=end_lr)
        learn.model.forward = learn.model.real_forward

        return learn

    def create_arch_from_decisions(self, decisions):
        f_d = decisions["filters"]
        g_d = decisions["groups"]
        k_d = decisions["kernels"]
        d_d = decisions["dilations"]
        a_d = decisions["activations"]
        st_d = decisions["strides"]
        # sk_d = decisions["skips"]

        f = self.filters
        g = self.groups
        k = self.kernels
        d = self.dilations
        a = self.activations
        st = self.strides
        # sk = self.skips

        arch = []
        # arch_skips = []
        arch_activations = []
        f_idx = f_d[0]

        for i in range(self.num_layers):
            g_idx = g_d[i]
            k_idx = k_d[i]
            d_idx = d_d[i]
            a_idx = a_d[i]
            st_idx = st_d[i]
            # sk_idx = sk_d[i]

            if i == 0:
                in_ch = self.CH
            else:
                # in_ch = f[f_d[i-1]]
                in_ch = f[f_idx]

            # f_idx = f_d[i]            

            groups = g[g_idx]
            out_channels = f[f_idx]

            groups = groups - (out_channels % groups)

            if i == 0:
                groups = 1
            else:
                if groups > in_ch:
                    if in_ch > out_channels:
                        groups = out_channels
                    else:
                        groups = in_ch
            try:
                padding = np.ceil(self.R*((st[st_idx]-1)/2))
            except IndexError as e:
                print(e)
                set_trace()
            
            padding += np.ceil(((k[k_idx]-1) + (k[k_idx]-1)*(d[d_idx]-1))/2)

            conv = nn.Sequential(*[nn.Conv2d(in_channels=in_ch, 
                            out_channels=out_channels, 
                            kernel_size=k[k_idx],
                            stride=st[st_idx],
                            dilation=d[d_idx],
                            groups=groups,
                            padding=padding),
                            nn.BatchNorm2d(out_channels)])
            arch.append(conv)
            # arch_skips.append(sk[sk_idx])
            arch_activations.append(a[a_idx])

        arch.append(
            nn.Sequential(*[
                nn.Linear(f[f_idx]*self.R*self.C, self.num_classes)
                , nn.LogSoftmax(dim=1)
            ])
        )

        class Arch(nn.Module):
            def __init__(self):
                super(Arch, self).__init__()

                self.arch = nn.ModuleList(arch)

            def forward(self, input):
                # skips = np.array(arch_skips).astype("float32")
                x = input
                layer_outputs = []
                for i, (layer, actv) in enumerate(zip(self.arch, arch_activations)):
                    # if i > 1:
                    #     skips = skips[:i]
                    #     skips_total = (1.0 * skips.sum())
                    #     if skips_total != 0:
                    #         skips /= skips_total
                    #         skip_attention = 0
                    #         for k, s in enumerate(skips):
                    #             skip_attention += layer_outputs[k]*float(s) 
                    #         x = (skip_attention + x) / 2
                    
                    x = actv(layer(x))
                    layer_outputs.append(x)

                x = x.view(x.shape[0], -1)

                out = self.arch[-1](x)

                return out

        return Arch()


        
