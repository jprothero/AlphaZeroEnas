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

        #specify the order of the decisions
        #one offs should probably be first
        self.decision_list = [
            "filters"
            , "groups"
            , "skips"
            , "kernels"
            , "dilations"
            , "activations"
            , "strides"
        ]

        #maybe want to have a layer embedding
        #so each layer has a unique embedding
        #we dont necessarily need to because we are doing a weighted mix of all previous ones
        #if we always use a layer embedding for that we wont have to except 
        #on filters, we can just always do some combination of the previous options
        #why not I guess

        #so we need a smarter way to integrate the embedding merges
        #basically we need one additional softmax
        #and the additional softmax can be separate for simplicity, i.e. an
        #embedding merge softmax

        self.embedding_merge_softmax = nn.Sequential(*[
                LayerNorm(lstm_size),
                nn.Linear(lstm_size, total_embeddings + num_layers), 
                nn.Softmax(dim=1)
            ])

        #and additional layer embeddings (optional)

        def skip_condition(layer_idx):
            #skip when layer index = 0 (no skip options)
            #or when layer_idx = 1 (only 1 option, doesnt make sense)
            #true means normal, false means skip this decision
            if layer_idx > 1:
                return True
            else:
                return False

        def filter_condition(layer_idx):
            #only do one filters choice then skip future ones
            if layer_idx == 0:
                return True
            else:
                return False

        self.decision_conditions = dict()

        for name in self.decision_list:
            if name is "skips":
                self.decision_conditions[name] = skip_condition
            elif name is "filters":
                self.decision_conditions[name] = filter_condition
            else:
                self.decision_conditions[name] = lambda x: True

        self.mask_conditions = dict()

        def skip_mask(layer_idx, probas):
            #so zero out the probas for all layers after the current one
            #so for example lets say we're on layer 2
            #we want to zero out the probability for 2-the end
            #that means we would want layer_idx-1 actually so we include 2
            probas[layer_idx-1:] *= 0
            return probas

        for name in self.decision_list:
            if name is "skips":
                self.mask_conditions[name] = skip_mask
            else:
                self.mask_conditions[name] = None

        # def filters_condition():
        #     if self.filters_times_used is None:
        #         self.filters_times_used = 1
        #     else:
        #         return False

        # def skips_condition():
        # self.conditions = [None for _ in range(len(self.decision_list))]
        # self.conditions[0] = lambda x: 

        #so lets see.. we could 

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

        #they can be accessed with len(embeddings) - num_layers - layer_idx
        #add an additional num_layers worth of embeddings
        embeddings.extend([nn.Parameter(torch.rand(lstm_size)-.5) for
                _ in range(num_layers)])

        #so how can we get these layer embeddings.
        #the start index would be len(embeddings) - num_layers - layer_idx
        #ugh just realized the emb merge thing 
        #mmmm 
        #sigh, idk this is hard to fix. I just have to change so much stuff to get this working
        #and it's hard to maintain simplicity
        #it would be better if we could keep things the same 

        self.softmaxs = nn.ModuleList(softmaxs)
        self.embeddings = nn.ParameterList(embeddings)

    def do_sim(self, az, cont_out):
        if az.curr_node["d"] == az.max_depth-1:
            return

        #so lets see, during the expand we are ignoring creating children
        #where the condition results in false
        #then the select should effectively ignore it
        #so I think we just need to modify the expand
        cont_out, decision_idx = az.select(self.starting_indices, self.decision_list,
            self.embeddings, self.controller, cont_out)

        depth = az.curr_node["d"]
        layer_idx = depth // self.num_layers
        decision_idx = depth % len(self.decision_list)
        decision_name = self.decision_list[decision_idx]
        condition = self.decision_conditions[decision_name]

        #so lets see... for skips for example we want to mask certain things, right?
        #we want to mask all layers > the current layer_idx

        #so a natural thing would be that we maybe have mask conditions too

        #if true proceed like normal
        if condition(layer_idx):
            probas = self.softmaxs[decision_idx](cont_out).squeeze()
            probas_np = probas.detach().data.numpy()

            if self.mask_conditions[decision_name] is not None:
                probas_np = self.mask_conditions[decision_name](layer_idx, probas_np)

            az.expand(probas_np, self.decision_conditions, self.decision_list)

            value = self.value_head(cont_out.squeeze())
            value = value.detach().data.numpy()
            az.backup(value)
        #else we continue on to the next simulation

    #so lets see.. we want the memories to be in a format which is easy for 
    #the net to train on
    #for that we want all of the results to be in a list, then a tensor
    #for the search probas we'd like to make them a tensor and do elementwise multiplication
    #so basically I want to get the memory in format where I can concat all the search probas
    #and the policies, and I can do all of the values around the same time
    #right now I can basically do that. one issue is that I need to load the whole trajectory
    #which is pretty time consuming. but as long as we are maintaining the autoregressive quality
    #it might be the best we can do. if we didnt have autoregression we could probably 
    #compute dynamically the unique embedding which is a combination of all previous embeddings,
    #I feel like ditching the autoregressiveness might be way to improve efficiency
    #we can use transformer attention or dilated convs
    #basically right now we have a unique embedding for each choice
    #and we have a unique softmax for each decision
    #and maybe we have a unique embedding for each layer?
    #that would allow us to more easily specify differences between layers, which is definitely
    #important. so basically we could concat learn an attention combine of all the embeddings,
    #so basically all the decisions chosen and all of the layer embeddings
    #basically we are trying to find a way to combine all of the different embeddings to create
    #a unique embedding which decodes to give instructions about what to do next
    #so we could do a masking thing maybe, like as we're going through the flow of the net
    #we mask all embeddings that havent been used yet
    #so for example we have a softmax (or hierarchical softmax / mixture) which 
    #looks over all of the different options creates a new unique embedding
    #so we want to create a new unique embedding for each step of the algo
    #i.e. we want to create a unique attentional combination of all previous embeddings
    #one issue with this is that if we choose the same embedding twice we will lose that 
    #notion. maybe we can weigh the embeddings based on the number of times they've been used
    #i.e. multiply each embedding in the softmax by the number of times it's been used

    #so basically we progressively mask all of the embeddings and use attention to combine
    #them into one new embedding which tells what to do next.

    #so for example we choose emb1, and we mask everything except emb1 and the start_emb
    #since the start_emb would be in everyhthing we can exclude it.

    #then we pick emb5, we would mask all but those two, and normalize the softmax between the two
    #(we may want to add a sharpness tanh or something, or possibly mix multiple softmaxs)

    #then we pick emb1 again(just for showing) and emb1 will get a weight of 2, while emb5 will
    #get a weight of 1, and the rest will get a weight of 0

    #I think this is nicer because we are getting away from expensive autoregressive stuff
    #and we can use training data much more efficiently, all we would need to store would be
    #the final normalized softmax and we can use that to create the embeddings and send them
    #to the net.

    #another idea is we can have multiple softmax heads and mix their results
    #that way we arent forced to do one mix of of the embeddings, we could do many mixes
    #so that would allow more control.
    #one question is how are we going to train these softmaxs that pick the embeddings
    #one obvious solution is to run it through alphazero also, and basically we would interweave
    #each decision with the embedding creation decision. it would effectively double the number of 
    #sims, but it would be a strong way to determine how we create the embedding, which I like a lot

    #I feel like it wouldn't be too hard to do either. we basically could add an option to select
    #if it's the embedding combiner softmax, and if so we multiply each of the policies 
    #by the mask. I guess that would be in the expand actually, when we create the policy
    #make the policy 0 for each of the masked options, and we pass the pass to teh function

    #so what to do first. I definitely dont want to forget to combine the search probas when
    #we evaluate them, but thats conceptually simple, it's just batching to reduce overhead
    #basically

    #I never really analyzed the alphazero loss, it basically says (if the search probas are
    # confident, your choice matters more, i.e. if the search probas are confident, the 
    # policy should be confident). now one issue it is that of course, the more confident the
    # algo gets, the closer other numbers will be to zero, which will result in a high loss for those
    # that isn't totally satisfactory. can we invert it or something? 
    #what do we want. we want the policy to match the search probas. there may be better ways
    #to do that, such as KL div, jensen, MSE, etc. the confidence thing here I feel is a bit
    #dubious for the reasons mentioned above, it will almost always have high loss
    #we could square or cube the search probas, so that the loss for ones that the search_probas
    #doesnt care about dont matter almost at all, which I feel is better, but idk,
    #kind of feels like a hack

    #and there is a lot of research out there about getting two probability distributions to match
    #it seems like jensen is one of the best ones, or what about wasserstein loss for example?
    #what if we do something like log of the difference between the two?
    #so .9 - .7 would be 

    #I guess it would be the inverse log
    #so 1 - (diff)
    #so for example (.9 - .7) = .2
    #log(.8) = small = good
    #so basically it would be the closer that the two are to each other the smaller 
    #the loss would be. 
    #now one obvious issue with is that we might not want the probabilty distribution to
    #exactly match. we might want just some gentle encouragement for what we care about
    #we could still do the weighting probably, where basically we mainly want to change
    #the good high confidence options

    #so the above -log(1 - (diff)) * search_prior**2 (for more sharpness)
    #I feel like that is pretty nice
    #can we matmul that?
    #yeah we can for sure

    #another thing we need to deal with is removing some redundant calculations
    #for example right now we 

    #for all expands we can add in a mask, or actually we can just do it from the policy!
    #for the policy we can just mask what we dont want, and the UCT will stay 0
    #so for example for the skips we can have it be over all layers between 1 and L-1
    #and then mask the ones that would be impossible at that time
    #and for the embedding combiner decisions, we can just multiple each of the probabilities
    #for each of the embeddings by their number of visits

    #another thing we need to fix is making it so that we can have one off decisions
    #which arent repeated. i.e. we want to make it so that decisions are skipped
    #under certain conditions. 
    #for example we want to skip layer skip options when there was only one previous layer,
    #because there is no choice to be made. 

    #and for outfilters for example, we need to pick whether we will allow different filter 
    #sizes or not. I would say not because it will in the long run give us a lot more 
    #flexibility by counting on the out filter sizes always being the same.

    #so where to start... for now I think removing the redundant calculations would be good
    #so what would having conditional decision skipping entail...
    #basically when we expand, we have a skip or dont skip option
    #and basically we could look up by the decision index, and evaluate a function
    #which tells us whether or not we should expand

    #now one obvious issue with this is that we're kind of complicating the currently
    #pretty simple loop
    #I think the mask for the expands is easy to add in
    #but injecting a bunch of code to make conditional skips is going to complicate things
    #a lot. 
    #it begs the question, maybe we should just add a section dedicated to one off runs
    #i.e. out filters wouldnt be in decisions, it would be in one_off_decisions
    #and we run all of the one off decisions at the beginning (or wherever)
    #and then we do the normal loop

    #I think thats better, because we really dont want to over complicate things.
    #simplicity I really feel is a key factor in having something clever

    #so lets see. basically we could add an initial "do first" option or something

    #ive been thinking about it more.. and it may be just as disruptive to change the training
    #loop. #probably I should just skip certain decisions if the conditions are met when we 
    #expand
    #so we have a list which links a decision index to a condition function
    #if the condition function is none we proceed like normal
    #otherwise we pass whatever info the condition function needs, 
    #and we will return whether we proceed like normal or skip to the next decision

    #btw I think mixture of softmaxs might be important
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
                "search_probas": torch.from_numpy(visits).float()
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
        #batch everything up where possible

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
                #we cant really batch the value loss while it's autoregressive
                #going to fix that soon....
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


        
