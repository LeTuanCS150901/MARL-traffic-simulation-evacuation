import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):  
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()

        # hidđen layers
        # layer_sizes = [obs_dim, 128, 128, 256, 128, 64, 32]
        layer_sizes = [obs_dim, 512, 512, 512, 512, 128, 128]
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
    
# import torch
# import torch.nn as nn

# class QNetwork(nn.Module):  
#     def __init__(self, obs_dim, action_dim):  
#         super(QNetwork, self).__init__()  

#         # toy net    
#         # self.hiddenlayer = nn.ModuleList([  
#         #     nn.Linear(obs_dim, 64),  
#         #     nn.Linear(64, 64),  
#         #     nn.Linear(64, 128),  
#         #     nn.Linear(128, 64),  
#         #     nn.Linear(64, 32)  
#         # ])  

#         # manhattan
#         # self.hiddenlayer = nn.ModuleList([
#         #     nn.Linear(obs_dim, 128),
#         #     nn.Linear(128, 128),
#         #     nn.Linear(128, 256),
#         #     nn.Linear(256, 128),
#         #     nn.Linear(128, 64),
#         #     nn.Linear(64, 32)
#         # ])

        
#         # manhatta with large number of agents
#         self.hiddenlayer = nn.ModuleList([
#             nn.Linear(obs_dim, 512),
#             nn.Linear(512, 512),
#             nn.Linear(512, 512),
#             nn.Linear(512, 128),
#             nn.Linear(128, 128),
#         ])
#         self.outputlayer = nn.Linear(128, action_dim)  

#         # Optional: Add sigmoid layer for [0, 1] output
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):  
#         # Apply sigmoid to constrain outputs to [0, 1]
#         #return self.sigmoid(self.outputlayer(x))
#         for hl in self.hiddenlayer:  
#             x = torch.relu(hl(x))  
#         return self.outputlayer(x)
#         # return self.sigmoid(self.outputlayer(x))
    
# if __name__ == "__main__":  
    
#     # Example usage  
#     obs_dim = 5  # Set obs_dim to 5 to match input length  
#     action_dim = 4  # Number of actions  

#     model = QNetwork(obs_dim, action_dim)  
#     print(model)  

#     # Example observation tensor (must be of shape [obs_dim])  
#     observation = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)  
#     print("Observation input:", observation)  

#     # Forward pass - outputs Q-values for all 4 actions  
#     q_values = model(observation)  
#     print("Q-values for all actions:", q_values)  

#     # Example feasible actions indices (subset of [0, action_dim-1])  
#     feasible_actions = torch.tensor([1, 2, 3])  
#     print("Feasible actions indices:", feasible_actions)  

#     # Masking q_values to only feasible actions  
#     q_values_masked = q_values[feasible_actions]  
#     print("Masked Q-values (only feasible actions):", q_values_masked)