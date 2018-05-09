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

from fastai.imports import *
from fastai.transforms import *
from fastai.learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from copy import deepcopy as dc

from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import ThreadPoolExecutor as TPE
from torch.multiprocessing import Pool, set_start_method, get_context
from torch.nn.utils import weight_norm as wn

from torch import optim

from .transformer import make_model

#so as of right now it takes a positional embedding

#so it takes a src_vocab, 
#makes it embedding size of d_model, such as 512
#the embedding has a positional encoding (somehow)
#and then that is fed into the encoder

#so it takes whatever the input is and embeddings it into 512 dimensions
#so for example it will embed a sentence of arbitrary length into 512 dimensions, I tihnk
#need to test

#so lets see, we want Embeddings to be 
#Embeddings(num_embeddings, model_dims)
#then it goes into the positional encoder -> 
#then that goes into the encoder -> 
#well... we have a couple different embeddings
#idk we can kind of bypass the embeddings, 
#so for example we pass in the set of embeddings and encode their positional 
#info, so like the trajectory, and run it through the PositionalEncoding
#then it goes into the encoder like normal
#now, we need the trajectory to be masked right?
#because it can be variable length.
#i.e. we just want 

#so the embedding takes in anything from size vocab and outputs that many samples
#in (num_in, emb_dims)
#and the question is, does the encoder take a fixed num_in, or is it batched?
#I kind of assume that it's fixed

#hmmm... could we alter this to do it over batches?
#that would be a heck of a lot nicer.
#but it would be variable length, so that's an issue

#sooo my understanding is that it basically does softmaxs over itself and 
#does multiplicative adding, like combines it's normal output with it
#so it effectively learns to focus over certain area of it's output
#since it is encoding a whole sentence worth, that makes sense.
#I mean, ideally what we want is probably to look at a whole possible trajectory
#but again, this locks us into certain things, which is not ideal

#so idk, for my ultimate goals I really want the system to be as dynamic as possible
#so for that I think I may need to stick to RNN's
#which is really annoying because I just switched away from them,
#but I think it's probably necessary. 
#this way I can have dynamic number of sims, etc

#I really do think that self attention and dilated convolutions are important though.
#maybe we can do self attention over the recurrence?

#I'm not exactly sure how we could do that. we cant do a fixed length softmax 
#because again, it's dynamic

#we could generate a logit at each time step and do a softmax over those

#alright, so lets go over what we want to accomplish

#I think one of the might issues with the net right now is the way that 
#embeddings are handled. 

#we need a set of embeddings, so we can just define a simple nn.Embedding

#thats fine,

#but then as we are going through the network we need those embeddings 
#to be remembered and to dictate the flow of our network.

#so as of right now I was trying to flatten all of the embeddings, then
#combine them somehow, such as having a softmax over all of the possible
#embeddings, and then masking ones that dont make sense

#I feel like its a little flawed

#and the issue with the transformer attention is that it takes fixed length sequences,
#whereas we have variable length ones,
#and we have limited

#honestly, we want a dynamic system that is as simple as specifying t and t+1
#for that I think we need some type of RNN
#maybe we can do an RNN and add the multi head attention between inner time steps
#i.e. dynamically generate X logits, which will form the multihead attention,
#and the input to the lstm will be a softmax + 
#and maybe we would need to add positional (temporal) encodings?

#so I feel like step one is probably adding back an LSTM

#idea: have different RNNs for different time scales:
#a trajectory/uct RNN, a readout RNN, and maybe more, such as a meta RNN
#each of those would be handling different scales of memory and would feed into each other
#that would probably eliminate the need for backup nets,
#then we would just enforce the MCTS formula from UCT using the 
#policy from the simulation RNN, then do a final strong move using the output
#or maybe the output of the simulation RNN could be the "history" and 
#that is used as input to a softmax which is the simulation_net

#and then the final history is fed into the readout RNN, which looks at histories,
#and makes a final prediction.
#and perhaps we can even have a more macro RNN, that takes the history from the readout_nets
#and uses that to update an episode wide, or long term memory.

#since in theory all of these RNN's are feeding into each other, maybe we could make them
#one RNN, and then zero out the output / change the chrono init at certain points
#need to see.

#also need to investigate memory over different time scales, and
#consider maybe switching the LSTM from being affine to using dilated convs or something

#I dont think it makes sense wor

# https://stackoverflow.com/questions/8277715/multiprocessing-in-a-pipeline-done-right
#good multiprocessing/pipeline resource

#idea:
#what if we did dilated convs, with self attention between recurrence,
#as a forget gate only lstm with chrono init (janet), and had it look
#over an entire program (itself)
#i.e. it could look at a whole program and predict a better one
#and the performance is measured some way.
#we maybe could do causal convs with masking or something,
#and maybe could make the recurrence to trajectories (imagine states)

#interesting idea

#two papers on chrono init
#https://arxiv.org/pdf/1804.11188.pdf
#https://arxiv.org/pdf/1804.04849.pdf

#Kanerva Machine another interesting memory resource,
#but its a bit more complicated, and I think the LSTM thing is simpler for now.
#https://arxiv.org/pdf/1804.01756v1.pdf

def init_lstm(lstm, hidden_size, T_max):
    for name, params in lstm.state_dict().items():
        if "weight" in name:
            nn.init.xavier_uniform_(params)
        elif "bias" in name:
            init = torch.log(torch.rand(hidden_size)*(T_max - 1) + 1)
            params[:hidden_size] = -init.clone()
            params[hidden_size:2*hidden_size] = init

    return lstm

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

#https://arxiv.org/pdf/1704.00325.pdf
#parallel MCTS paper, good resource

def lambda_identity(x): return x
def lambda_false(x): return False
def filter_condition(layer_idx): return False if layer_idx == 0 else True
def skip_condition(layer_idx): return False if layer_idx > 1 else True
def skip_mask(layer_idx, probas):
    probas[layer_idx-1:] = 0
    probas /= (1.0 * probas.sum())
    return probas

class ENAS(nn.Module):
    def __init__(self, num_classes=10, R=32, C=32, CH=3, num_layers=4, controller_dims=70, 
            num_controller_layers=5, value_head_dims=70, num_value_layers=5, cuda=torch.cuda.is_available(),
            num_fastai_batches=20):
        super(ENAS, self).__init__()
        self.num_classes = num_classes
        self.R = R
        self.C = C
        self.CH = CH
        self.num_layers = num_layers
        self.lstm_size = controller_dims
        self.num_controller_layers = num_controller_layers
        self.has_cuda = cuda
        
        self.controller = nn.LSTM(controller_dims, controller_dims, num_controller_layers)

        self.fake_data = self.create_fake_data(num_fastai_batches)

        value_layers = []

        for _ in range(num_value_layers):
            value_layers.extend([
                nn.Linear(controller_dims, value_head_dims)
            , LayerNorm(value_head_dims)
            , nn.Tanh()
            ])
           
        value_layers.extend([
            nn.Linear(value_head_dims, 1)
            , nn.Tanh()
            ])

        self.value_head = nn.Sequential(*value_layers)

        self.filters = [
            16,
            32,
        ]

        self.groups = []

        i = 0
        while True:
            self.groups.append(2**i)
            if 2**i == self.filters[-1]:
                break
            i += 1

        self.groups = list(reversed(self.groups))

        self.kernels = [
            1,
            3,
            5,            
        ]

        self.dilations = [
            1,
            2,
            4,
        ]

        self.activations = [
            nn.Softmax(dim=1),
            nn.Tanh(),
            nn.Sigmoid(),
            nn.ReLU(),
            lambda_identity
        ]

        self.strides = [
            1,
            2,
            3
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

        self.decision_list = [
            "filters"
            , "groups"
            , "skips"
            , "kernels"
            , "dilations"
            , "activations"
            , "strides"
        ]

        #the number of decisions is 1 + 2 + 4*5
        self.num_decisions = len(self.decision_list)*(self.num_layers) - \
            (self.num_layers - 2) - (self.num_layers - 1)

        self.controller = init_lstm(self.controller, controller_dims, self.num_decisions)

        self.controller.flatten_parameters()

        self.decision_conditions = dict()

        for name in self.decision_list:
            if name is "skips":
                self.decision_conditions[name] = skip_condition
            elif name is "filters":
                self.decision_conditions[name] = filter_condition
            else:
                self.decision_conditions[name] = lambda_false

        self.mask_conditions = dict()

        for name in self.decision_list:
            if name is "skips":
                self.mask_conditions[name] = skip_mask
            else:
                self.mask_conditions[name] = None

        self.starting_indices = []

        self.first_emb = nn.Parameter(torch.rand(controller_dims)-.5)#.view(1, 1, -1)
        softmaxs = []
        embeddings = []
        starting_idx = 0
        for name in self.decision_list:
            options = self.decisions[name]
            softmaxs.append(nn.Sequential(*[
                LayerNorm(controller_dims),
                nn.Linear(controller_dims, len(options)), 
            ]))

            embeddings.extend([nn.Parameter(torch.rand(controller_dims)-.5) for
                _ in range(len(options))])

            self.starting_indices.append(starting_idx)
            starting_idx += len(options)

        self.softmaxs = nn.ModuleList(softmaxs)
        self.embeddings = nn.ParameterList(embeddings)

        min_layers = []
        max_layers = []

        min_settings = {}
        max_settings = {}

        for key, val in self.decisions.items():
            min_settings[key] = val[0]
            max_settings[key] = val[-1]
            
        for _ in range(self.num_layers):
            min_layers.append(min_settings)
            max_layers.append(max_settings)
            
        min_arch = self.create_arch_from_settings(min_layers)
        max_arch = self.create_arch_from_settings(max_layers)

        self.min_params = self.count_parameters(min_arch)
        self.max_params = self.count_parameters(max_arch)

        del min_arch
        del max_arch

    def scale_by_parameter_size(self, x):
        return (x - self.min_params) / (self.max_params - self.min_params)

    def check_condition(self, az, layer_idx, decision_name):
        condition = self.decision_conditions[decision_name]
        return condition(layer_idx)

    def get_values(self, alpha_zeros):
        cont_outs = []
        for az in alpha_zeros:
            cont_outs.append(az.cont_out)

        cont_outs = torch.cat(cont_outs)

        values = self.value_head(cont_outs.squeeze())

        for az, value in zip(alpha_zeros, values):
            az.value = value.detach().item()

        return az

    def go_to_root(self, az):
        while az.curr_node["parent"] is not None:
            az.curr_node = az.curr_node["parent"]
        return az

    def backup(self, az):
        az.backup(az.value)
        return az

    def expand(self, az):
        probas = az.probas
        hidden = az.hidden

        depth = az.curr_node["d"]
        decision_idx = depth % len(self.decision_list)
        decision_name = self.decision_list[decision_idx]
        layer_idx = depth // len(self.decision_list)

        if self.mask_conditions[decision_name] is not None:
            probas = self.mask_conditions[decision_name](layer_idx, probas)

        if az.do_expand:
            az.expand(probas, hidden)

        return az

    def evaluate(self, alpha_zeros):
        decision_indices = []

        decision_indices_lists = [[] for _ in range(len(self.decision_list))]

        for i, az in enumerate(alpha_zeros):
            decision_indices_lists[az.decision_idx].append(i)
        
        if len(alpha_zeros[0].trajectory) > 0:
            embeddings = [self.embeddings[az.trajectory[-1]].unsqueeze(0) for az in alpha_zeros]
            
            embeddings = torch.cat(embeddings).unsqueeze(0)
        else:
            embeddings = [self.first_emb.unsqueeze(0) for _ in alpha_zeros]
            embeddings = torch.cat(embeddings).unsqueeze(0)
        
        if alpha_zeros[0].hidden is not None:
            hs = []
            cs = []

            for az in alpha_zeros:
                hs.append(az.hidden[0])
                cs.append(az.hidden[1])

            hs = torch.cat(hs, dim=1)
            cs = torch.cat(cs, dim=1)
            hidden = (hs, cs)
        else:
            hidden = None

        cont_outs, (hs, cs) = self.controller(embeddings, hidden)
        cont_outs = cont_outs.squeeze(0)
        hiddens = [(hs[:, i, :].unsqueeze(1), 
            cs[:, i, :].unsqueeze(1)) for i in range(hs.shape[1])]

        for az, hidden in zip(alpha_zeros, hiddens):
            az.hidden = hidden

        for i, decision_indices in enumerate(decision_indices_lists):
            if len(decision_indices) > 0: 
                specific_cont_outs = cont_outs[decision_indices]
                logits = self.softmaxs[i](specific_cont_outs).view(len(specific_cont_outs), -1)
                probas = F.softmax(logits, dim=1)
                azs = [alpha_zeros[i] for i in decision_indices]
                for az, p in zip(azs, probas):
                    az.probas = p.squeeze().detach().data
                    if self.has_cuda:
                        az.probas = az.probas.cpu()
                    
                    az.probas = az.probas.numpy()

        for az, cont_out in zip(alpha_zeros, cont_outs):
            az.cont_out = cont_out.unsqueeze(0)
            if az.curr_node["d"] < az.max_depth:
                az.do_expand = True
            else:
                az.do_expand = False
        
    def simulate(self, az):
        trajectory = az.select(self.starting_indices, self.decision_list)

        depth = az.curr_node["d"]

        layer_idx = depth // len(self.decision_list)
        decision_idx = depth % len(self.decision_list)
        decision_name = self.decision_list[decision_idx]

        while True:
            skip_curr = self.check_condition(az, layer_idx, decision_name)
            if not skip_curr:
                break
            else:
                az.curr_node["d"] += 1
                depth = az.curr_node["d"]
                layer_idx = depth // len(self.decision_list)
                decision_idx = depth % len(self.decision_list)
                decision_name = self.decision_list[decision_idx]

        az.trajectory = trajectory
        az.decision_idx = decision_idx

        return az

    def get_memories(self, az):
        az.new_memories[-1]["decisions"] = az.decisions
        # for memory in az.new_memories:
        #     memory["decisions"] = az.decisions

        return az.new_memories

    def reset_to_root(self, az):
        while az.curr_node["parent"] is not None:
            az.curr_node = az.curr_node["parent"]

        return az
    
    def move_choice(self, az):
        d = az.curr_node["d"]
        
        decision_idx = d % len(self.decision_list)
        starting_idx = self.starting_indices[decision_idx]
        name = self.decision_list[decision_idx]
        choice_idx, visits = az.select_real() 

        emb_idx = starting_idx + choice_idx
        az.decisions[name].append(choice_idx)

        az.new_memories.append({
            "search_probas": torch.from_numpy(visits).float()
            , "trajectory": dc(az.real_trajectory)
            , "decision_idx": decision_idx
        })

        az.real_trajectory.append(emb_idx)
        az.trajectory = az.real_trajectory

        if d < az.max_depth-1:
            az.done = False
        else:
            az.done = True

        return az

    def make_architecture_mp(self, kwargs):
        num_archs, num_sims = kwargs["num_archs"], kwargs["num_sims"]
        
        alpha_zeros = [AlphaZero(max_depth=self.num_layers*len(self.decision_list)) for _ in range(num_archs)]

        decisions = dict()
        for name in self.decision_list:
            decisions[name] = []

        for az in alpha_zeros:
            az.real_trajectory = []
            az.decisions = dc(decisions)
            az.new_memories = []

        del decisions

        final_alpha_zeros = []

        i = 0
        while True:
            print(f"Choice {i} of {self.num_decisions-1}")
            # start = datetime.datetime.now()
            for _ in range(num_sims):
                alpha_zeros = list(map(self.simulate, alpha_zeros))

                self.evaluate(alpha_zeros)

                # with TPE(max_workers) as executor:

                alpha_zeros = list(map(self.expand, alpha_zeros))

                self.get_values(alpha_zeros)

                # with TPE(max_workers) as executor:
                alpha_zeros = list(map(self.backup, alpha_zeros))

                for az in alpha_zeros:
                    assert az.curr_node["parent"] is None

                # with PPE(max_workers) as executor:
                #     alpha_zeros = list(executor.map(self.reset_to_root, alpha_zeros))
                # if j % num_sims == num_sims-1:
                #     end = datetime.datetime.now()
                #     difference = end - start
                #     # print(difference.seconds)
                #     print(difference.microseconds / (1.0 * 1e6)) 

            # with PPE(max_workers) as executor:
            #     alpha_zeros = list(executor.map(self.reset_to_root, alpha_zeros))

            # for az in alpha_zeros:
            #     assert az.curr_node["parent"] is None

            not_done_alpha_zeros = []
            for az in alpha_zeros:
                az = self.move_choice(az)
                if az.done:
                    final_alpha_zeros.append(az)
                else:
                    not_done_alpha_zeros.append(az)

            alpha_zeros = not_done_alpha_zeros

            i += 1
            
            if len(alpha_zeros) == 0:
                break

        # with TPE(max_workers) as executor:
        new_memories = list(map(self.get_memories, final_alpha_zeros))

        #so I am returning a list of memories
        #[memories, memories]

        return new_memories

    def create_fake_data(self, num_batches=20):
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

        num_samples = 64*num_batches
        trn_X = np.zeros(shape=(num_samples, self.CH, self.R, self.C))
        trn_y = np.zeros(shape=(num_samples, 1))
        val_X = np.zeros(shape=(num_samples//6, self.CH, self.R, self.C))
        val_y = np.zeros(shape=(num_samples//6, 1))
        trn = [trn_X, trn_y]
        val = [val_X, val_y]
        fake_data = ImageClassifierData.from_arrays("./data", trn=trn, val=val,
                                        classes=classes)
        return fake_data

    def forward(self, X):
        pass

    def get_min_parameters(self):
        pass

    #got score of .484, pretty high
    #{'filters': [1], 'groups': [3, 3, 3, 3], 'skips': [0, 1], 
    # 'kernels': [0, 0, 0, 0], 'dilations':[2, 2, 1, 2], 'activations': [2, 2, 2, 2], 
    # 'strides': [2, 2, 2, 2]}

    def train_controller(self, _=None, __=None):
        batch = sample(self.memories, self.batch_size)

        search_probas = []
        policies = []
        values = []
        scores = []

        value_loss = 0

        search_probas = []
        values = []
        scores = []

        for i, memory in enumerate(batch):
            trajectory = memory["trajectory"]
            score = memory["score"]
            decision_idx = memory["decision_idx"]
            
            cont_out, hidden = self.controller(self.first_emb.view(1, 1, -1))

            for emb_idx in trajectory:
                emb = self.embeddings[emb_idx].view(1, 1, -1)
                cont_out, hidden = self.controller(emb, hidden)
                value = self.value_head(cont_out).view(-1)
                values.append(value)
                scores.append(score)
            logits = self.softmaxs[decision_idx](cont_out).squeeze()
            probas = F.softmax(logits.unsqueeze(0), dim=1).squeeze()
            policies.append(probas)
            search_probas.append(memory["search_probas"])
                
        scores = torch.tensor(scores).float()
        values = torch.cat(values).float()

        if self.has_cuda:
            scores = scores.cuda()
            values = values.cuda()

        value_loss = F.mse_loss(values, scores)

        search_probas = torch.cat(search_probas)
        policies = torch.cat(policies)

        if self.has_cuda:
            search_probas = search_probas.cuda()
            policies = policies.cuda()

        search_probas_loss = -search_probas.unsqueeze(0).mm(torch.log(policies.unsqueeze(-1)))
        search_probas_loss /= self.batch_size

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Value mean: ", values.mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Dist diff mean: ", torch.abs(search_probas - policies).mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        if self.has_cuda:
            print(f"Probas: {search_probas_loss.data.cpu().squeeze().numpy()}, Value {value_loss.data.cpu().numpy()}")
        else:
            print(f"Probas: {search_probas_loss.squeeze().data.numpy()}, Value {value_loss.data.numpy()}")
            
        total_loss = search_probas_loss + value_loss 
        # total_loss = dist_matching_loss
        # total_loss = value_loss

        return total_loss

    def train_controller_batched(self, _=None, __=None):
        batch = sample(self.memories, self.batch_size)

        search_probas = []
        policies = []
        values = []
        scores = []

        # decision_indices = []
        trajectory_indices = [[] for _ in range(self.num_decisions)]
        batch_indices = [[] for _ in range(self.num_decisions)]
        done_indices = [[] for _ in range(self.num_decisions+1)]
        not_done_indices = [[] for _ in range(self.num_decisions+1)]
        done_decision_indices = [[] for _ in range(self.num_decisions+1)]
        done_search_probas = [[] for _ in range(self.num_decisions+1)]

        value_loss = 0
        for i, memory in enumerate(batch):
            trajectory = memory["trajectory"]
            if len(trajectory) > 0:
                not_done_indices[0].append(i)
                for j, emb_idx in enumerate(trajectory):
                    trajectory_indices[j].append(emb_idx)
                    batch_indices[j].append(i)
                    j += 1
                    not_done_indices[j].append(i)
                del not_done_indices[j]

                done_indices[j].append(i)
                done_decision_indices[j].append(memory["decision_idx"])
                done_search_probas[j].append(memory["search_probas"][0])
            else:
                del not_done_indices[0]
                done_indices[0].append(i)
                done_decision_indices[0].append(memory["decision_idx"])
                done_search_probas[0].append(memory["search_probas"][0])
            #what I want to do is have a list that has all of the done indices
            #so basically if we have any indices ending on that iteration
            #we do the search probas loss
            #so what we need is a list of the done indices for 
            #each of the iterations through the trajectories

            score = memory["score"]
            # decision_idx = memory["decision_idx"]
            # decision_indices.append(decision_idx)

            scores.append(score)

        search_probas = []
        scores = torch.tensor(scores)

        if self.has_cuda:
            scores = scores.cuda()

        value_loss = 0

        embeddings = [self.first_emb.unsqueeze(0) for _ in range(self.batch_size)]
        embeddings = torch.cat(embeddings).unsqueeze(0)
        if self.has_cuda:
            embeddings = embeddings.cuda()

        cont_outs, hidden = self.controller(embeddings)
        cont_outs = cont_outs.squeeze(0)

        def do_search_probas_part(idx):
            for cont_out, decision_idx in zip(cont_outs[done_indices[idx]], 
                done_decision_indices[idx]):

                logits = self.softmaxs[decision_idx](cont_out).squeeze()
                probas = F.softmax(logits.unsqueeze(0), dim=1).squeeze()

                policies.append(probas)
                search_probas.append(done_search_probas[idx])

        if len(done_indices[0]) > 0:
            do_search_probas_part(0)

        hidden = (hidden[0][:, not_done_indices[0], :], hidden[1][:, not_done_indices[0], :])
        done_search_probas = done_search_probas[1:] 
        done_decision_indices = done_decision_indices[1:] 
        done_indices = done_indices[1:]
        not_done_indices = not_done_indices[1:]

        for i, emb_indices in enumerate(trajectory_indices):
            embeddings = []
            for idx in emb_indices:
                embeddings.append(self.embeddings[idx].unsqueeze(0))
            if len(embeddings) == 0:
                break

            embeddings = torch.cat(embeddings).unsqueeze(0)
            if self.has_cuda:
                embeddings = embeddings.cuda()
                
            cont_outs, hidden = self.controller(embeddings, hidden)
            cont_outs = cont_outs.permute(1, 0, 2)
            values = self.value_head(cont_outs).squeeze()

            value_loss += F.mse_loss(values, scores[batch_indices[i]].float())
            if len(done_indices[i]) > 0:
                do_search_probas_part(i)
                hidden = (hidden[0][:, not_done_indices[i], :], hidden[1][:, not_done_indices[i], :])
                if len(hidden[0].shape) < 3:
                    break
        value_loss /= self.num_decisions

        set_trace()
        search_probas = torch.cat(search_probas)

        if self.has_cuda:
            search_probas = search_probas.cuda()

        policies = torch.cat(policies)

        if self.has_cuda:
            policies = policies.cuda()

        search_probas_loss = -search_probas.unsqueeze(0).mm(torch.log(policies.unsqueeze(-1)))
        search_probas_loss /= self.batch_size

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Value mean: ", values.mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Dist diff mean: ", torch.abs(search_probas - policies).mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        if self.has_cuda:
            print(f"Probas: {search_probas_loss.data.cpu().squeeze().numpy()}, Value {value_loss.data.cpu().numpy()}")
        else:
            print(f"Probas: {search_probas_loss.squeeze().data.numpy()}, Value {value_loss.data.numpy()}")
            
        total_loss = search_probas_loss + value_loss 
        # total_loss = dist_matching_loss
        # total_loss = value_loss

        return total_loss

    def fastai_train(self, controller, memories, batch_size, num_cycles=10, epochs=1, min_memories=None):
        self.memories = memories
        self.batch_size = batch_size
        if min_memories is None:
            min_memories = batch_size*30

        if (len(memories) < min_memories):
            print("Have {} memories, need {}".format(len(memories), min_memories))
            return
        controller_wrapped = FastaiWrapper(model=controller, crit=self.train_controller)
        controller_learner = Learner(data=self.fake_data, models=controller_wrapped)
        controller_learner.crit = controller_wrapped.crit
        controller_learner.opt_fn = optim.Adam
        controller_learner.model.train()

        controller_learner.model.real_forward = controller_learner.model.forward

        controller_learner.model.forward = lambda x: x
        controller_learner.fit(4e-2, epochs, wds=1e-5) #was 7e-2
        # controller_learner.fit(2, epochs, cycle_len=num_cycles, use_clr_beta=(10, 13.68, 0.95, 0.85), 
        #     wds=1e-4)

        controller_learner.model.forward = controller_learner.model.real_forward

        del self.memories
        del self.batch_size
        del controller_learner.model.real_forward
        del controller_learner
    #so let me think for a sec, the issue is in theory more filters will always be better at the cost of more memory and compute
    #so we should divide score by number of filters and multiply by the max number of filters, so it will be between 1/max_filters and
    #max_filters/max_filters = 1

    #actually no, what we want is that the score is divide by between 1 and max_filters
    #so basically we would be maximizing score per filter
    #which, in theory it is going to be lower
    #i.e. probably it would find the lowest number of filters that didnt cause the model to fail, which is ideal

    def controller_lr_find(self, controller, memories, batch_size, start_lr=1e-5, end_lr=2):
        self.memories = memories
        self.batch_size = batch_size
        if (len(memories) < batch_size):
            print("Have {} memories, need {}".format(len(memories), batch_size))
            return
        arch = FastaiWrapper(model=controller, crit=self.train_controller)

        arch.model.real_forward = arch.model.forward
        arch.model.forward = lambda x: x

        learn = Learner(data=self.fake_data, models=arch)
        learn.opt_fn = optim.Adam
        learn.crit = arch.crit
        learn.model.train()

        learn.lr_find(start_lr=start_lr, end_lr=end_lr)
        learn.model.forward = learn.model.real_forward

        return learn

    #sooo I want the min num filters and the max number filters
    #min = num_layers times filters[0] and max = num_layers * filters[-1]
    #and then basically we count up the filters as we make the architecture
    #and make a scaling factor based on that.
    #then one issue is that unless 32 filters is 100% better than 16 filters, it will probably
    #end up always choosing 16 filters. that isn't necessarily bad, but we want a strong 
    #what we could do is scale the scaler, i.e. make it something like 1 - (alpha*scaler)
    #and that would mean that we only change the score a little bit based on the filters
    #that way for example, if the difference is only a score of 10%, we wouldnt want that, but it it's any more
    #than that, we would prefer the smaller number of filters
    def create_arch_from_decisions(self, decisions):
        f_d = decisions["filters"]
        g_d = decisions["groups"]
        k_d = decisions["kernels"]
        d_d = decisions["dilations"]
        a_d = decisions["activations"]
        st_d = decisions["strides"]
        sk_d = decisions["skips"]

        f = self.filters
        g = self.groups
        k = self.kernels
        d = self.dilations
        a = self.activations
        st = self.strides
        sk = self.skips

        arch = []
        arch_skips = []
        arch_activations = []
        f_idx = f_d[0]

        for i in range(self.num_layers):
            g_idx = g_d[i]
            k_idx = k_d[i]
            d_idx = d_d[i]
            a_idx = a_d[i]
            st_idx = st_d[i]

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
            if i > 1:
                sk_idx = sk_d[i-2]
                arch_skips.append(sk[sk_idx])
            arch_activations.append(a[a_idx])

        arch.append(
            nn.Sequential(*[
                nn.Linear(f[f_idx]*self.R*self.C, self.num_classes)
                , nn.LogSoftmax(dim=1)
            ])
        )

        has_cuda = self.has_cuda

        class Arch(nn.Module):
            def __init__(self):
                super(Arch, self).__init__()

                self.has_cuda = has_cuda
                self.arch = nn.ModuleList(arch)

            def forward(self, input):
                skips = np.array(arch_skips).astype("float32")
                # skips = torch.tensor(arch_skips).float()
                # if self.has_cuda:
                #     skips = skips.cuda()
                x = input
                layer_outputs = []
                for i, (layer, actv) in enumerate(zip(self.arch, arch_activations)):
                    if i > 1:
                        skips = skips[:i]
                        skips_total = (1.0 * skips.sum())
                        if skips_total != 0:
                            skips /= skips_total
                            skip_attention = 0
                            for k, s in enumerate(skips):
                                skip_attention += layer_outputs[k]*float(s) 
                            x = (skip_attention + x) / 2
                    
                    x = actv(layer(x))
                    layer_outputs.append(x)

                x = x.view(x.shape[0], -1)

                out = self.arch[-1](x)

                return out

        return Arch()

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def create_arch_from_settings(self, settings):
        arch = []
        arch_skips = []
        arch_activations = []

        for i, layer_settings in enumerate(settings):
            filters = layer_settings["filters"]
            groups = layer_settings["groups"]
            kernels = layer_settings["kernels"]
            dilations = layer_settings["dilations"]
            activations = layer_settings["activations"]
            strides = layer_settings["strides"]
            skips = layer_settings["skips"]

            if i == 0:
                in_ch = self.CH
            else:
                in_ch = filters

            out_channels = filters

            groups = groups - (out_channels % groups)

            if i == 0:
                groups = 1
            else:
                if groups > in_ch:
                    if in_ch > out_channels:
                        groups = out_channels
                    else:
                        groups = in_ch
            padding = np.ceil(self.R*((strides-1)/2))
            
            padding += np.ceil(((kernels-1) + (kernels-1)*(dilations-1))/2)

            conv = nn.Sequential(*[nn.Conv2d(in_channels=in_ch, 
                            out_channels=out_channels, 
                            kernel_size=kernels,
                            stride=strides,
                            dilation=dilations,
                            groups=groups,
                            padding=padding),
                            nn.BatchNorm2d(out_channels)])

            arch.append(conv)
            if i > 1:
                arch_skips.append(skips)

            arch_activations.append(activations)

        arch.append(
            nn.Sequential(*[
                nn.Linear(filters*self.R*self.C, self.num_classes)
                , nn.LogSoftmax(dim=1)
            ])
        )

        has_cuda = self.has_cuda

        class Arch(nn.Module):
            def __init__(self):
                super(Arch, self).__init__()

                self.has_cuda = has_cuda
                self.arch = nn.ModuleList(arch)

            def forward(self, input):
                skips = np.array(arch_skips).astype("float32")
                # skips = torch.tensor(arch_skips).float()
                # if self.has_cuda:
                #     skips = skips.cuda()
                x = input
                layer_outputs = []
                for i, (layer, actv) in enumerate(zip(self.arch, arch_activations)):
                    if i > 1:
                        skips = skips[:i]
                        skips_total = (1.0 * skips.sum())
                        if skips_total != 0:
                            skips /= skips_total
                            skip_attention = 0
                            for k, s in enumerate(skips):
                                skip_attention += layer_outputs[k]*float(s) 
                            x = (skip_attention + x) / 2
                    
                    x = actv(layer(x))
                    layer_outputs.append(x)

                x = x.view(x.shape[0], -1)

                out = self.arch[-1](x)

                return out

        return Arch()

    def arch_lr_find(self, arch, data, start_lr=1e-5, end_lr=10):
        arch = FastaiWrapper(model=arch, crit=None)
        learn = Learner(data=data, models=arch)
        learn.crit = F.nll_loss
        learn.opt_fn = optim.Adam
        learn.lr_find(start_lr=start_lr, end_lr=end_lr, wds=1e4)
        return learn

               



        
