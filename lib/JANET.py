from torch import nn
import torch
import torch.nn.functional as F
from copy import deepcopy

from ipdb import set_trace

#interesting related papers:
#https://arxiv.org/pdf/1804.01756v1.pdf (Kanerva Memory)
#https://arxiv.org/pdf/1804.11188.pdf (technical)
#https://arxiv.org/pdf/1804.04849.pdf (JANET)

#.2 dropout is good
#wider nets seem a bit better
#progressive growing,
#maybe could apply to embedding size or something,
#and could maybe do concat of the average and max pooling for each?
#also the fastai has a .half() to make the net faster
class JANET(nn.Module):
    def __init__(self, in_dims, h_dims, T_max, num_layers, activation=True, add_layer_norm=True):
        super(JANET, self).__init__()
        layers = []

        for i in range(num_layers):
            layer_in = in_dims if i == 0 else h_dims
            layers.append(JanetLayer(layer_in, h_dims, T_max, activation))
            
            if add_layer_norm:
                layers.append(LayerNorm(h_dims))
            
        layers.append(nn.Linear(h_dims, 10))
        layers.append(nn.LogSoftmax(dim=-1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class JanetLayer(nn.Module):
    def __init__(self, in_dims, h_dims, T_max, activation):
        super(JanetLayer, self).__init__()
        if activation is not None:
            self.activation = nn.PReLU()
        
        self.U_f = nn.init.xavier_uniform_(nn.Parameter(torch.rand(h_dims, h_dims)))
        self.W_f = nn.init.xavier_uniform_(nn.Parameter(torch.rand(in_dims, h_dims)))
        self.b_f = nn.Parameter(torch.log(torch.rand(h_dims)*(T_max - 1) + 1))

        self.U_h = nn.init.xavier_uniform_(nn.Parameter(torch.rand(h_dims, h_dims)))
        self.W_h = nn.init.xavier_uniform_(nn.Parameter(torch.rand(in_dims, h_dims)))
        self.b_h = nn.Parameter(torch.zeros(h_dims))

        self.last_h = None

    def forward(self, sequences):
        #Expect shape (sequence_length, batch_size, segment_length)
        #ex: mnist would be (784, batch_size, 1)
        #I think keeping the last_h here is probably okay, I dont see why not
        #sooo lets see... the input size is going to be the segment length
        
        hs = []
        for x in sequences:
            if self.last_h is None:
                f_t = F.sigmoid(x @ self.W_f + self.b_f)
                h_t = (1 - f_t) * F.tanh(x @ self.W_h + self.b_h)
            else:
                f_t = F.sigmoid(self.last_h @ self.U_f + x @ self.W_f + self.b_f)
                h_t = f_t * self.last_h + (1 - f_t) * F.tanh(self.last_h @ self.U_h \
                    + x @ self.W_h + self.b_h)
                
               
            if self.activation is not None:
                h_t = self.activation(h_t)
               
            self.last_h = h_t
            hs.append(h_t)
            
        hs = torch.cat(hs)

        return hs