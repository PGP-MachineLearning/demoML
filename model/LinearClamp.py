import torch
import torch.nn as nn

class LinearClamp(nn.Module):
    
    def __name__(self) -> str:
        return "LinearClamp"
    
    
    def forward(self, x):
        return torch.clamp(x, 0, 1)