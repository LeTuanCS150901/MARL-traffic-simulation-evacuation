import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()

        # hidđen layers
        # toy net
        # layer_sizes = [obs_dim, 128, 128, 256, 128, 64, 32]
        
        # manhattan network
        # layer_sizes = [obs_dim, 512, 512, 512, 512, 128, 128]

        # manhattan2 big network
        layer_sizes = [obs_dim, 512, 1024, 1024, 1024, 512, 256]
        self.hiddenlayer = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])

        # LayerNorm after each Linear (except output)
        self.bn_layers = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])
        # output layer
        self.outputlayer = nn.Linear(layer_sizes[-1], action_dim)  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):  
        # hidden -> batch -> relu
        for hl, bn in zip(self.hiddenlayer, self.bn_layers):  
            x = hl(x)
            x = bn(x)
            x = torch.relu(x)  
        return self.outputlayer(x)
    