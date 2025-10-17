import torch
import torch.nn as nn

class Model(nn.Module):
    
    def __init__(self,weight_path : str | None, description : str, hidden_sizes:  list[int], hidden_activation = nn.ReLU, output_activation = nn.Sigmoid, use_mask = True) -> None:
        super(Model, self).__init__()  
        
        lst = [item for sublist in [[nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), hidden_activation() if i < len(hidden_sizes)-1 else output_activation()] for i in range(len(hidden_sizes)-1)] for item in sublist]
        self.inner = nn.ModuleList(lst)
        self.mask = nn.Parameter(torch.ones(hidden_sizes[0]), requires_grad=True) if use_mask else torch.ones(hidden_sizes[0])
        self.sigmoid = nn.Sigmoid()
        
        self.description = description
        if weight_path:
            self.load_state_dict(torch.load(weight_path))
            
        
        
    def forward(self, x):
        x = x * self.sigmoid(self.mask)
        for layer in self.inner:
            x = layer(x)
        x = x.squeeze()
        return x