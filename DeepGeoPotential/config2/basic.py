import math
import torch
from torch import nn
import random
class Linear(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(Linear,self).__init__()
        self.linear = nn.Linear(dim_in,dim_out)
    def forward(self,x):
        x = self.linear(x)
        return x


class LinearNoBias(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(LinearNoBias,self).__init__()
        self.linear = nn.Linear(dim_in,dim_out,bias=False)
    def forward(self,x):
        x = self.linear(x)
        return x
    



