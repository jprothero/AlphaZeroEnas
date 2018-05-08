class LSTMClassifier(nn.Module):
    def __init__(self, seq_len=784, in_size=1, hidden_size=100, num_layers=4):
        super(LSTMClassifier, self).__init__()
        seq_len //= in_size
        
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_size = in_size
        
        self.importance_logits = []
        self.mixing_logits = []
        self.lstm_layers = []
        for i in range(num_layers):
            lstm_in = in_size if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(lstm_in, hidden_size, 1))
            
            self.importance_logits.append(nn.Linear(hidden_size*self.in_size, 1))
            self.mixing_logits.append(nn.Linear(hidden_size*self.in_size, 1))
            
            for name, params in self.lstm_layers[-1].state_dict().items():
                if "weight" in name:
                    wn(self.lstm_layers[-1], name)
                    nn.init.xavier_uniform_(params)
                elif "bias" in name:
                    init = nn.Parameter(torch.log(torch.rand(hidden_size)*(sequence_len - 1) + 1))
                    params[:hidden_size] = -init
                    params[hidden_size:2*hidden_size] = init
                    
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

        self.lin = nn.Sequential(*[
            nn.Linear(hidden_size*sequence_len, 10)
            , nn.LogSoftmax(dim=-1)
        ])
        
        #so whta do I want to do
        #I want the output of each individual layer to output how important it thinks it is
        
    def forward(self, sequence):
        batch_size = sequence.shape[1]
        hidden_importance_logits = torch.zeros(self.num_layers, self.in_size, self.seq_len)
        output_importance_logits = torch.zeros(self.num_layers, self.in_size, self.seq_len)
        
        #so lets see, hiddens should be num_layers (1), 1, hidden_size?
        #I tihnk?
        hs = torch.zeros(self.num_layers, 1, self.in_size, self.hidden_size)
        cs = torch.zeros(self.num_layers, 1, self.in_size, self.hidden_size)
        
        hidden = None
        for j, x in enumerate(sequence):
            mixing_logits = torch.zeros(self.num_layers)
            x = x.unsqueeze(0)
            for i, (layer, importance_logit, mixing_logit) in enumerate(zip(self.lstm_layers, 
                                                          self.importance_logits, self.mixing_logits)):
                if j > 0:
                    set_trace()
                    hidden = (hs[i, 0], cs[i, 0])
                    
                if i > 1:
                    mixing_probas = F.sigmoid(mixing_logits[i-1])
                    set_trace()
                    hidden_importance_probas = F.softmax(hidden_importance_logits[:i], dim=-1)
                    layer_importance_probas = F.softmax(hidden_importance_logits[:i], dim=-1)
                    
                    h = hs[:i].clone()
                    c = cs[:i].clone()
                    
                    hidden = (h*importance_probas, c*importance_probas)
                    hidden = (hs[i-1]*(1-mixing_probas) + mixing_probas*hidden[0].sum(), 
                              cs[i-1]*(1-mixing_probas) + mixing_probas*hidden[1].sum())
                    
                    x = mixing_probas*hidden[0]
                    
                    #hmmm so lets see...
                    #we're updating the hidden for the previous layer kind of?
                    #because we're basically taking all of the 
                    #so lets see..
                    #like what we're doing is taking all of the importance_logits and hiddens (outs)
                    #for the previous of the layer, and we're ...
                    #I think we can simplify this alot
                    #we can probably make it lists
                    #or keeping it as a tensor is fine
                    #and we want 
                    
                    
                x, (h, c) = layer(x, hidden)
                
                hs[i] = h
                cs[i] = c
                
                #so x should be equal to h
                #so the shape will be (1, 1, self.hidden_size, self.in_size)
                #so the importance_logit will take 
                #nvm the x is the output from the inputs
                #the h is the weight matrix or whatever
                
                #so x will be (batch_size, self.hidden_size, self.in_size)
                #so the importance_logit should take hidden_size, 
                importance_logits[i, :, j] = importance_logit(x)
                mixing_logits[i] = mixing_logit(x)
                
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)
        log_probas = self.lin(x)
        
        return log_probas