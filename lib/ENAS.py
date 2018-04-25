import torch.nn as nn
from .qrnn import QRNN
from . import FLAGS
from torch.nn import functional as F
from torch.distributions import Categorical
import torch
import numpy as np
from torch.autograd import Variable
from ipdb import set_trace
from .CreateChild import build_model
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

class SimpleENAS(nn.Module):
    def __init__(self, num_classes=10, R=32, C=32, CH=3, num_layers=4, lstm_size=70, 
            num_lstm_layers=4, value_head_dims=64):
        super(SimpleENAS, self).__init__()
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

        # self.skips = [i for i in range(num_layers-1)]

        self.decisions = {
            "filters": self.filters,
            "groups": self.groups,
            "kernels": self.kernels,
            "dilations": self.dilations,
            "activations": self.activations,
            "strides": self.strides,
            # "skips": self.skips
        }

        self.decision_list = [
            "filters"
            , "groups"
            , "kernels"
            , "dilations"
            , "activations"
            , "strides"
            # , "skips"
        ]

        # def filters_condition():
        #     if self.filters_times_used is None:
        #         self.filters_times_used = 1
        #     else:
        #         return False

        # def skips_condition():
        # self.conditions = [None for _ in range(len(self.decision_list))]
        # self.conditions[0] = lambda x: 

        self.starting_indices = []

        self.first_emb = nn.Parameter(torch.rand(lstm_size)-.5).view(1, 1, -1)
        softmaxs = []
        embeddings = []
        #one solution is to have a list which gives the starting index
        #for a given decision_idx
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

        self.softmaxs = nn.ModuleList(softmaxs)
        self.embeddings = nn.ParameterList(embeddings)

    def do_sim(self, az, cont_out):
        if az.curr_node["d"] == az.max_depth-1:
            return

        cont_out, decision_idx = az.select(self.starting_indices, self.decision_list,
            self.embeddings, self.controller, cont_out)

        probas = self.softmaxs[decision_idx](cont_out).squeeze()
        probas_np = probas.detach().data.numpy()
        az.expand(probas_np)

        value = self.value_head(cont_out.squeeze())
        value = value.detach().data.numpy()
        az.backup(value)

    def make_architecture(self, num_sims=14):
        self.filter_chosen = False
        new_memories = []
        az = AlphaZero(max_depth=self.num_layers*len(self.decision_list))

        cont_out = self.controller(self.first_emb)[0].squeeze(0)

        orig_cont_out = cont_out.clone()

        choice_indices = []
        decision_indices = []

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

            #so we are coming in with the root node d = -1
            d = az.curr_node["d"]
            decision_idx = d % len(self.decision_list)
            choice_idx, visits = az.select_real() 
            #we pick the most visited child (stochastically)
            #and d is now = 0
            #d % number of decisions so that it loops around 

            #choose name for decision 0
            name = self.decision_list[decision_idx]

            # len_v = len(visits)
            # len_d = len(self.decisions[name])
            # set_trace()
            # assert len(visits) == len(self.decisions[name])

            #so I could have the decisions up to this point, or the list of
            #choice_indices and decisions_indices to this point
            #that would allow having many different samples
            #that actually seems like a great idea.
            choice_indices.append(choice_idx)
            decision_indices.append(decision_idx)

            new_memories.append({
                "search_probas": visits
                , "choice_indices": c(choice_indices)
                , "decision_indices": c(decision_indices)
            })

            decisions[name].append(choice_idx)

            starting_idx = self.starting_indices[decision_idx]

            #starting_idx is when it starts, + choice_idx determines which for that
            emb = self.embeddings[starting_idx + choice_idx].view(1, 1, -1)

            orig_cont_out = self.controller(emb)[0]
            cont_out = orig_cont_out.clone()

            if d == az.max_depth-1:
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
            probas = probas.unsqueeze(-1)
            search_probas = Variable(torch.from_numpy(search_probas).float())
            search_probas = search_probas.unsqueeze(0)
            search_probas_loss += -search_probas.mm(torch.log(probas))

            search_probas_loss /= len(search_probas) #might be wrong
            # value_loss /= len(choice_indices)

        value_loss /= self.batch_size
        search_probas_loss /= self.batch_size
        value_loss *= 6
        # value_loss *= 200
        # value_loss *= 150 #this works pretty good, but the search_probas
        # search_probas_loss /= self.batch_size
        # search_probas_loss /= 28
        # print("Value loss: {}, Search_probas loss: {}".format(value_loss.data.numpy(),
        #     search_probas_loss.data.numpy())) 
        # # total_loss = value_loss/80 + search_probas_loss*160 #seems to work good
        # # total_loss = value_loss/160 + search_probas_loss*320 #okay too, value slow
        # total_loss = value_loss/80 + search_probas_loss*160
        # total_loss = value_loss + search_probas_loss
        # total_loss = #value_loss + 
        # print(f"Value: {value_loss.data.numpy()}, Probas {search_probas_loss.data.numpy()}")
        total_loss = search_probas_loss + value_loss

        #probas: .2-.4, maybe .35
        #value: idk it doesnt even make sense

        # self.validating = not self.validating

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
        # controller_learner.fit(.2, 1) #.33
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
        #so let me think for a sec, I would like to make the net work as easily as possible
        #i.e. not fail over and over (separable for example would error out for like 99.9%
        # of choices)
        #dilations and strides will change the r and c but we can pad them back to the originals
        #out_filters will change the out_channels but we can progressively track it
        #skips will increase the in_channels, but we should be able to keep track of it 
        #and account for it

        #sooo how I have this right now it would be kind of hard to make certain
        #things only happen once in the loop
        #how would we do that?
        #right now we loop around and give each thing a decision
        #we could maybe have something where we skip over a decision if we don't care about it
        #lets try to do it, because it will give efficiency and more control
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

            f_idx = f_d[i]            

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

            padding = np.ceil(self.R*((st[st_idx]-1)/2))
            
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


        
