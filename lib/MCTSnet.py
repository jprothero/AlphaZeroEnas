import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

import model_configs

from IPython.core.debugger import set_trace

def conv_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)

#sooooo lets see... 
#what can we improve from MCTSnet
#I feel like the main ideas are that we are using an external memory
#to inform our decisions about what we're doing
#and that memory is going to be reset at the end of doing a number of sims
#i.e. the end of a trajectory probably

#as of right now the MCTSnet flow is
#emb_net(state) -> embedding -> uct_net(embedding) -> leaf_node 
#so to add alpha zero to it, we need to have each policy also have a value,
#i.e. a prediction of what it thinks the current state it is at will be

#so basically we train the embeddings, and the backup net, based on minimizing the 
#error for the value and the policy

#so let me just think about what we need to do. 
#we should start simple, get MCTS net working and trainable, and then proceed from there.
#so we somehow embed an input state into an embedding space for the network.
#then we use the hash of that embedding? or no, the hash of that state
#to identify if we've seen it before

#issue: for example with the ENAS where we are using a pool of embeddings,
#that really isn't relevant
#so instead of using an embedding net, the states are embeddings, and
#we refer to them by their decision idx's

#ideally the goal is we make a drop in replacement for our current controller which
#instead uses MCTSnet concepts

#so basically we want to keep what we have with the 

class PolicyBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims, add_gate=True, gate_only=False):
        super(PolicyBlock, self).__init__()
        self.add_gate = add_gate

        self.conv1 = conv_layer(in_dims, h_dims)
        self.bn1 = nn.BatchNorm2d(h_dims)
        self.relu = nn.ReLU()
        if out_dims is None:
            out_dims = h_dims
        self.conv2 = conv_layer(h_dims, h_dims)
        self.bn2 = nn.BatchNorm2d(h_dims)

        self.policy = nn.Linear(h_dims*6*7, out_dims)

        if self.add_gate:
            self.gate = nn.Linear(in_dims*6*7, out_dims)
            self.tanh = nn.Tanh()

def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = self.bn1(x)
    x += residual
    x = self.relu(x)
    
    x = self.conv2(x)
    x = self.bn2(x)
    x += residual
    x = self.relu(x)

    out = F.softmax(self.policy(x.view(config.BATCH_SIZE, -1)), dim=1)
    
    if self.add_gate:
        out *= self.tanh(self.gate(x.view(config.BATCH_SIZE, -1)))

    return out


class ResBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=None, add_gate=False, gate_only=False):
        super(ResBlock, self).__init__()
        self.add_gate = add_gate
        self.gate_only = gate_only

        self.conv1 = conv_layer(in_dims, h_dims)
        self.bn1 = nn.BatchNorm2d(h_dims)
        self.relu = nn.ReLU()
        if out_dims is None:
            out_dims = h_dims
        self.conv2 = conv_layer(h_dims, out_dims)
        self.bn2 = nn.BatchNorm2d(out_dims)
        if add_gate:
            self.tanh = nn.Tanh()
            self.conv3 = conv_layer(h_dims, out_dims)
            self.bn3 = nn.BatchNorm2d(out_dims)

    def forward(self, x):
        residual = x.squeeze()
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.add_gate:
            g = self.conv3(out)
            g = self.bn3(g)
            g = self.tanh(g)
            if self.gate_only:
                return g
            out = g*out

        out += residual
        out = self.relu(out)

        return out

class ValueHead(nn.Module):
    """This network does a readout... (update this)"""

    def __init__(self, in_dims, h_dims, out_dims=1, add_gate=False, gate_only=False):
        super(ValueHead, self).__init__()
        self.add_gate = add_gate

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.conv1 = nn.Conv2d(in_planes, num_filters, kernel_size=1, stride=1)
        self.lin1 = nn.Linear(in_dims, h_dims)
        self.bn1 = nn.BatchNorm1d(h_dims)
        self.lin2 = nn.Linear(h_dims, h_dims)
        if self.add_gate:
            self.tanh = nn.Tanh()
            self.gate = nn.Linear(h_dims, out_dims)
        self.lin3 = nn.Linear(h_dims, out_dims)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        if self.add_gate:
            gated = self.tanh(self.gate(x))
        x = self.lin3(x)
        x = self.tanh(x)
        
        if self.add_gate:
            x *= gated
        
        return x

def run_simulations(self, joint_states, curr_player, turn):
    self.embeddings = dict()
    S = dict()
    A = dict()
    R = dict()
    H = dict()
    N = dict()
    game_over = False
    memory = {
        "curr_player": curr_player,
        "result": None,
        "policy": {
            "output": []
        },
        "readout": {
            "output": None
        },
        "value": {
            "output": None
        }
    }

    root_state = np.concatenate((np.expand_dims(joint_states[0], 0),
                            np.expand_dims(joint_states[1], 0),
                            np.zeros(shape=np.expand_dims(joint_states[1], 0).shape) + curr_player), axis=0)

    t = 0
    #+1 sims since the first is used to expand the embedding
    for sim in range(config.MCTS_SIMS+1):
        while True:
            try:
                N[hashed] += 1
            except:
                N[hashed] = 0
                break

            legal_actions = self.get_legal_actions(S[t][:2])

            reward, game_over = self.calculate_reward(S[t][:2])

            R[t] = reward
            if len(legal_actions) == 0 or game_over:
                game_over = True
                break

            # consider moving the value head here and using it in the backups
            action = self.simulate(self.embeddings[hashed], S[t],
                                    sim, memory)

            A[t] = action

            new_state = self.transition(
                np.copy(S[t][:2][curr_player]), A[t])
            S[t + 1] = np.copy(S[t])
            S[t + 1][curr_player] = np.copy(new_state)
            t += 1
            curr_player += 1
            curr_player = curr_player % 2
            S[t][2] = curr_player
            S[t].flags.writeable = False
            hashed = hash(S[t].data.tobytes())
            S[t].flags.writeable = True

        if not game_over and len(legal_actions) > 0:
            state_one = cast_to_torch(S[t][0], self.cuda).unsqueeze(0)
            state_two = cast_to_torch(S[t][1], self.cuda).unsqueeze(0)
            state_three = cast_to_torch(S[t][2], self.cuda).unsqueeze(0)
            state = torch.cat(
                [state_one, state_two, state_three], 0).unsqueeze(0)
            self.models["emb"].eval()
            H[t] = self.embeddings[hashed] = self.models["emb"](state)

        if t > 0:
            H = self.backup(H, R, S, t, memory)
            t = 0

    self.models["readout"].eval()

    logits = self.models["readout"](H[0])

    memory["readout"]["output"] = logits

    pi = self.correct_policy(logits, joint_states, is_root=False)

    return pi, memory

def apply_temp_to_policy(self, pi, turn, T=config.TAU):
    if turn == config.TURNS_UNTIL_TAU0 or T == 0:
        temp = np.zeros(shape=pi.shape)
        temp[np.argmax(pi)] = 1
        pi = temp

    return pi

def simulate(self, emb, joint_state, sim, memory):
    emb = emb.view(1, 1, 8, 16)
    self.models["policy"].eval()
    logits, value = self.models["policy"](emb)

    if sim == 1:
        is_root = True
    else:
        is_root = False
    pi = self.correct_policy(logits, joint_state, is_root=is_root)

    idx = np.random.choice(len(self.actions), p=pi)

    action = self.actions[idx]
    memory["policy"]["output"].append({
        "log_action_prob": F.log_softmax(logits, dim=0)[idx], 
        "value": value, 
        "is_root": is_root
    })

    return action

def backup(self, H, R, S, _t, memory, is_for_inp=False):
    for t in reversed(range(_t)):
        reward = cast_to_torch([R[t]], self.cuda)
        comb_state_1 = S[t + 1][0] + S[t + 1][1]
        comb_state_2 = S[t][0] + S[t][1]
        action = comb_state_1 - comb_state_2
        action = cast_to_torch(action, self.cuda).view(-1)

        inp = torch.cat([H[t], H[t + 1], reward, action], 0)

        self.models["backup"].eval()
        H[t] = self.models["backup"](inp, H[t])

    return H

def correct_policy(self, logits, joint_state, is_root):
    odds = np.exp(logits.data.numpy())
    policy = odds / np.sum(odds)
    if is_root:
        nu = np.random.dirichlet([config.ALPHA] * len(self.actions))
        policy = policy * (1 - config.EPSILON) + nu * config.EPSILON

    mask = np.zeros(policy.shape)
    legal_actions = self.get_legal_actions(joint_state[:2])
    mask[legal_actions] = 1
    policy = policy * mask

    pol_sum = (np.sum(policy) * 1.0)
    if pol_sum == 0:
        return policy
    else:
        return policy / pol_sum

    return policy