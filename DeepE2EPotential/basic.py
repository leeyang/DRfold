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
    


def transform(k,rotation,translation):
    # K L x 3
    # rotation
    return torch.matmul(k,rotation) + translation


def batch_transform(k,rotation,translation):
    # k:            L 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('ba,bad->bd',k,rotation) + translation

def batch_atom_transform(k,rotation,translation):
    # k:            L N 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('bja,bad->bjd',k,rotation) + translation[:,None,:]

def IPA_transform(k,rotation,translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3
    return torch.einsum('bija,bad->bijd',k,rotation)+translation[:,None,None,:]

def IPA_inverse_transform(k,rotation,translation):
    # k:            L d1, d2, 3
    # rotation:     L 3 x 3
    # translation:  L 3   
    return torch.einsum('bija,bad->bijd',k-translation[:,None,None,:],rotation.transpose(-1,-2))

def update_transform(t,tr,rotation,translation):
    return torch.einsum('bja,bad->bjd',t,rotation),torch.einsum('ba,bad->bd',tr,rotation) +translation


def quat2rot(q,L):
    scale= ((q**2).sum(dim=-1,keepdim=True)    +1) [:,:,None]
    u=torch.empty([L,3,3],device=q.device)
    u[:,0,0]=1*1+q[:,0]*q[:,0]-q[:,1]*q[:,1]-q[:,2]*q[:,2]
    u[:,0,1]=2*(q[:,0]*q[:,1]-1*q[:,2])
    u[:,0,2]=2*(q[:,0]*q[:,2]+1*q[:,1])
    u[:,1,0]=2*(q[:,0]*q[:,1]+1*q[:,2])
    u[:,1,1]=1*1-q[:,0]*q[:,0]+q[:,1]*q[:,1]-q[:,2]*q[:,2]
    u[:,1,2]=2*(q[:,1]*q[:,2]-1*q[:,0])
    u[:,2,0]=2*(q[:,0]*q[:,2]-1*q[:,1])
    u[:,2,1]=2*(q[:,1]*q[:,2]+1*q[:,0])
    u[:,2,2]=1*1-q[:,0]*q[:,0]-q[:,1]*q[:,1]+q[:,2]*q[:,2]
    return u/scale








    
