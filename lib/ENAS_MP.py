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

from torch import optim

# https://stackoverflow.com/questions/8277715/multiprocessing-in-a-pipeline-done-right
#good multiprocessing/pipeline resource

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
            num_fastai_batches=8):
        super(ENAS, self).__init__()
        self.num_classes = num_classes
        self.R = R
        self.C = C
        self.CH = CH
        self.num_layers = num_layers
        self.lstm_size = controller_dims
        self.num_controller_layers = num_controller_layers
        self.has_cuda = cuda
        
        self.validating = False
        
        controller_layers = []
        for _ in range(num_controller_layers):
            controller_layers.extend([
            nn.Linear(controller_dims, controller_dims)
            , LayerNorm(controller_dims)
            , nn.Tanh()])

        self.controller = nn.Sequential(*controller_layers)

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

        # self.max_depth = 0
        # for key, _ in self.decisions.items():
        #     if key is "filters":
        #         self.max_depth += 1
        #     elif key is "skips": 
        #         self.max_depth += self.num_layers - 2
        #     else:
        #         self.max_depth += self.num_layers

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
                LayerNorm(controller_dims),
                nn.Linear(controller_dims, self.total_embeddings), 
            ])

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

        embeddings.extend([nn.Parameter(torch.rand(controller_dims)-.5) for
                _ in range(num_layers)])

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

    def scale_by_parameter_size(self, x):
        return (x - self.min_params) / (self.max_params - self.min_params)

    def check_condition(self, az, layer_idx, decision_name):
        condition = self.decision_conditions[decision_name]
        return condition(layer_idx)

    def embedding_from_trajectory(self, az=None, trajectory=None):
        if trajectory is None:
            trajectory = az.trajectory

        if len(trajectory) > 0:
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
            weights_list = []
            for _, idx in enumerate(indices):
                embeddings.append(self.embeddings[idx].unsqueeze(0))
                if len(trajectory) == 1:
                    return embeddings[-1].view(1, 1, -1)
                logits.append(self.emb_merge_pre_softmax(self.embeddings[idx])[indices].unsqueeze(0))
                weights_list.append(weights[idx])

            #this part isn't really parallelizable/batchable very easily
            #so let me think about this for a bit.
            #what the capsnets offer maybe is an alternative to conv nets
            #it also is an interesting target for an ENAS, since maybe we can design a better
            #or more efficient routing system or something.
            #but it is good to keep in mind that they will probably continue to get more efficient
            #for what I care about right now, which is trying to get ENAS with alpha zero working,
            #it doesnt really help. it could maybe help for the mcts gan idea or whatever,
            #but again capsnets are fairly experimental. we can consider it
            #but for now I think the trajectory is get_mp working -> get gpu working ->
            #see if we can make anything interesting / get it to train -> rent a paperspace
            #and test is more extensively then depending on the performance we can put
            #the project on the backburner, or do some improvements, such as trying to do
            #MCTSnet trained with alpha zero

            weights = torch.from_numpy(np.array(weights_list)).float().unsqueeze(-1)
            if self.has_cuda:
                weights = weights.cuda()
            weights = weights.view(1, -1)
            logits = torch.cat(logits)

            probas = F.softmax(logits, dim=1)
            probas = weights.mm(probas)

            # probas2 = F.softmax(logits)*weights
            # probas2 = probas2.sum(0)

            # assert (probas == probas2).all()

            probas /= probas.sum() #normalize
            probas = probas.view(1, -1)
            embeddings = torch.cat(embeddings)
            final_emb = probas.mm(embeddings)

            emb = final_emb.view(1, 1, -1)
        else:
            emb = self.first_emb.view(1, 1, -1)

        return emb

    def get_values(self, alpha_zeros):
        cont_outs = []
        for az in alpha_zeros:
            cont_outs.append(az.cont_out)

        cont_outs = torch.cat(cont_outs)

        values = self.value_head(cont_outs.squeeze())

        for az, value in zip(alpha_zeros, values):
            az.value = value.detach().item()

        return az

    def backup(self, az):
        az.backup(az.value)
        return az

    def expand(self, az):
        probas = az.probas

        depth = az.curr_node["d"]
        decision_idx = depth % len(self.decision_list)
        decision_name = self.decision_list[decision_idx]
        layer_idx = depth // len(self.decision_list)

        if self.mask_conditions[decision_name] is not None:
            probas = self.mask_conditions[decision_name](layer_idx, probas)

        az.expand(probas)

        return az

    def evaluate(self, alpha_zeros):
        # trajectories = []
        decision_indices = []

        decision_indices_lists = [[] for _ in range(len(self.decision_list))]

        for i, az in enumerate(alpha_zeros):
            # trajectories.append(az.trajectory)
            decision_indices_lists[az.decision_idx].append(i)
        
        # with PPE(self.max_workers) as executor:
        #     embeddings = list(executor.map(self.embedding_from_trajectory, alpha_zeros))
        embeddings = [self.embedding_from_trajectory(az) for az in alpha_zeros]

        embeddings = torch.cat(embeddings)

        cont_outs = self.controller(embeddings)

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
            az.cont_out = cont_out
        
    def simulate(self, az):
        if az.curr_node["d"] > az.max_depth-1: #was >=
            return az

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
        az.orig_emb = self.embedding_from_trajectory(az)

        if d < az.max_depth-1:
            az.done = False
        else:
            az.done = True

        #hmmm.... can we only return one?
        #like what if we only return done ones. 
        #but the issue is how do we differentiate which ones are or are not done
        #I guess we could probably return done ones here, and not done ones in the next step
        #then update.

        return az

    def make_architecture_mp(self, kwargs):
        # ctx = get_context("forkserver")
        num_archs, num_sims = kwargs["num_archs"], kwargs["num_sims"]
        
        alpha_zeros = [AlphaZero(max_depth=self.num_layers*len(self.decision_list)) for _ in range(num_archs)]

        decisions = dict()
        for name in self.decision_list:
            decisions[name] = []

        for az in alpha_zeros:
            az.orig_emb = self.first_emb
            az.real_trajectory = []
            az.decisions = dc(decisions)
            az.new_memories = []

        del decisions

        final_alpha_zeros = []

        i = 0
        while True:
            print(f"Choice {i} of {az.max_depth-1 - 5}")
            # start = datetime.datetime.now()
            for j in range(num_sims):
                # print(f"Sim {j}")
                # with TPE(max_workers) as executor:
                
                alpha_zeros = list(map(self.simulate, alpha_zeros))

                # with ctx.Pool() as executor:
                #     alpha_zeros = list(executor.map(self.simulate, alpha_zeros))

                # if j > 0:
                #     set_trace()
                self.evaluate(alpha_zeros)

                # with TPE(max_workers) as executor:

                alpha_zeros = list(map(self.expand, alpha_zeros))

                self.get_values(alpha_zeros)

                # with TPE(max_workers) as executor:
                alpha_zeros = list(map(self.backup, alpha_zeros))

                # for az in alpha_zeros:
                #     assert az.curr_node["parent"] is None

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

    def train_controller(self, _=None, __=None):
        batch = sample(self.memories, self.batch_size)

        search_probas = []
        policies = []
        values = []
        scores = []

        #we could speed this up like by batching the cont_outs
        #we can try it later

        #the training is supppppeeerrrr slow
        #it must be because we aren't using batching

        #well lets start adding it

        # device = torch.device("cuda" if self.has_cuda else "cpu")

        embeddings = []
        # embeddings = torch.zeros(self.batch_size, 1, self.controller_dims, device=device).float()
        decision_indices = []

        value_loss = 0
        for i, memory in enumerate(batch):
            trajectory = memory["trajectory"]
            score = memory["score"]
            sp = memory["search_probas"]
            decision_idx = memory["decision_idx"]
            decision_indices.append(decision_idx)

            # if len(trajectory) == 0:
            #     # cont_out = self.controller(self.first_emb)[0].squeeze(0)
            #     cont_out = self.controller(self.first_emb)
            # else:
            embedding = self.embedding_from_trajectory(trajectory=trajectory)
            embeddings.append(embedding)

            # cont_out = self.controller(embedding)

            #this is wrong but at some point we need to do something like this
            #we need to scale the score by the parameters value if the value is associated with increased
            #memory or especially compute. for example max groups might always be the best performing but
            #has much more compute reqs, so we want to minimize that, and similarly for filters we want 
            #to minimize the number of filters
            #we can probably come up with a more clever heuristic but for now / by number of filters would be okay.
            #we would learn the minimal number of filters where the information doesnt collapse basically (in theory)
            # if self.decision_list[decision_idx] is "filters":
            #     set_trace()
            #     score /= self.filters[trajectory[-1]]

            scores.append(score)
            # if self.training:
            #     self.value_head.register_hook(print)
            # value = self.value_head(cont_out.squeeze())
            # value_loss += F.mse_loss(value, score)
            # values.append(value)
            # logits = self.softmaxs[decision_idx](cont_out).squeeze()
            # probas = F.softmax(logits.unsqueeze(0), dim=1).squeeze()

            # policies.append(probas)
            search_probas.append(sp)
        embeddings = torch.cat(embeddings)
        search_probas = torch.cat(search_probas)
        scores = torch.tensor(scores)
        
        # search_probas = Variable(search_probas)

        if self.has_cuda:
            search_probas = search_probas.cuda()
            embeddings = embeddings.cuda()
            scores = scores.cuda()

        cont_outs = self.controller(embeddings)

        for cont_out, decision_idx in zip(cont_outs, decision_indices):
            logits = self.softmaxs[decision_idx](cont_out).squeeze()
            probas = F.softmax(logits.unsqueeze(0), dim=1).squeeze()

            policies.append(probas)

        values = self.value_head(cont_outs).squeeze()
            
        # values = torch.cat(values)
        # # if self.training:
        # #     values.register_hook(print)

        value_loss = F.mse_loss(values, scores)
        policies = torch.cat(policies)

        search_probas_loss = -search_probas.unsqueeze(0).mm(torch.log(policies.unsqueeze(-1)))
        search_probas_loss /= self.batch_size
        #/self.batch_size

        # values += 1
        # values /= 2

        # scores += 1
        # scores /= 2

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Value mean: ", values.mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        # values.register_hook(print)

        # value_loss = -ones.mm(torch.log(1 - torch.abs(scores - values).unsqueeze(-1)))

        # value_div = 1e6
        # dist_div = 5e2  #5e3 is good

        # value_loss = -torch.log(1 - torch.abs(scores - values)).sum()
        # ones = torch.ones(len(scores)).unsqueeze(0)        
        # value_loss = -ones.mm(torch.log(1 - torch.abs(scores - values)).unsqueeze(-1))
        # value_loss /= value_div         
        # value_loss /= 10       

        # policies = policies.unsqueeze(-1)
        # distance_from_one = 1 - torch.abs(search_probas - policies)
        # search_probas = search_probas.unsqueeze(0)**2
        # distance_from_one = distance_from_one.unsqueeze(-1)

        #issue: search_rpboas and policies may be result in negatives
        #can we tak the derivative of abs?we can try
        # dist_matching_loss = -(search_probas**2).unsqueeze(0).mm(torch.log(1 - \
        #  torch.abs(search_probas - policies).unsqueeze(-1)))
        # dist_matching_loss = -torch.log(1 - \
        #  torch.abs(search_probas - policies).unsqueeze(-1)).sum()
        # ones = torch.ones(len(search_probas)).unsqueeze(0)                

        # dist_matching_loss = -ones.mm(torch.log(1 - \
        #  torch.abs(search_probas - policies).unsqueeze(-1)))

        # dist_matching_loss = F.mse_loss(policies, search_probas)/(self.batch_size*len(policies))
        

        # print("*"*10)
        # print("*"*10)
        # print("*"*10)
        # print("Dist diff mean: ", torch.abs(search_probas - policies).mean())
        # print("*"*10)
        # print("*"*10)
        # print("*"*10)

        # dist_matching_loss = -(search_probas**2).unsqueeze(0).mm(torch.log(1 - \
        #  torch.abs(search_probas - policies).unsqueeze(-1)))

        # dist_matching_loss /= dist_div    
        # dist_matching_loss /= self.batch_size
        # dist_matching_loss /= self.batch_size
        # dist_matching_loss /= self.batch_size
        # dist_matching_loss /= self.batch_size
        # dist_matching_loss /= self.batch_size

        # dist_matching_loss = F.mse_loss(policies, search_probas)

        # dist_matching_loss /= len(search_probas) #might be wrong

        # search_probas_loss /= self.batch_size
        # if self.has_cuda:
        #     print(f"Dist: {dist_matching_loss.data.cpu().numpy()*dist_div}, Value {value_loss.data.numpy()*value_div}")
        # else:
        #     print(f"Dist: {dist_matching_loss.data.numpy()*dist_div}, Value {value_loss.data.numpy()*value_div}")

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
        controller_learner.model.train()

        controller_learner.model.real_forward = controller_learner.model.forward

        controller_learner.model.forward = lambda x: x
        controller_learner.fit(7e-2, epochs, wds=1e-6)
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

    def controller_lr_find(self, controller, memories, batch_size, start_lr=1e-5, end_lr=10):
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

               



        
