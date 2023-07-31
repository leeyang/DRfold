import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import basic



class MSAEncoder(nn.Module):
    def __init__(self,msa_dim,m_dim,z_dim):
        super(MSAEncoder,self).__init__()
        self.msalinear=basic.Linear(msa_dim,m_dim)
        self.qlinear  =basic.Linear(msa_dim,z_dim)
        self.klinear  =basic.Linear(msa_dim,z_dim)
        self.slinear  =basic.Linear(msa_dim,m_dim)
        self.pos = self.compute_pos2d()
        self.pos1d=self.compute_pos1d()
        self.poslinear=basic.Linear(129,z_dim)
        self.poslinear2=basic.Linear(14,m_dim)
    def tocuda(self,device):
        self.to(device)
        self.pos.to(device)
    def compute_pos1d(self,maxL=800):
       
        d = torch.arange(maxL)
        m = 14
        d =(((d[:,None] & (1 << np.arange(m)))) > 0).float()
        return d

    def compute_pos2d(self,maxL=800):
        a = torch.arange(maxL)
        b = (a[None,:]-a[:,None]).clamp(-64,64)
        return F.one_hot(b+64,129).float()


    def forward(self,msa):
        # msa should be masked
        if self.pos.device != msa.device:
            self.pos = self.pos.to(msa.device)
        if self.pos1d.device != msa.device:
            self.pos1d = self.pos1d.to(msa.device)
        # msa N L D, seq L D
        N,L,D=msa.shape
        s = self.slinear(msa[0])
        m = self.msalinear(msa)
        p = self.poslinear2(self.pos1d[:L])
        m = m + s[None,:,:] + p[None,:,:]
        sq=self.qlinear(msa[0])
        sk=self.klinear(msa[0])
        z=sq[None,:,:]+sk[:,None,:]
        z = z + self.poslinear( self.pos[:L,:L]  )
        return m,z

class SSEncoder(nn.Module):
    def __init__(self,ss_dim,z_dim):
        # ss_dim should be masked
        super(SSEncoder,self).__init__()
        self.qlinear  =basic.Linear(ss_dim,z_dim)
    def forward(self,ss):
        return self.qlinear(ss)

def fourier_encode_dist(x, num_encodings = 20, include_self = True):
    # from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch.py
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

class RecyclingEmbedder(nn.Module):
    def __init__(self,m_dim,z_dim,dis_encoding_dim):
        super(RecyclingEmbedder,self).__init__()  
        self.linear = basic.Linear(dis_encoding_dim*2+1,z_dim)
        self.dis_encoding_dim=dis_encoding_dim
        self.normz = nn.LayerNorm(z_dim)
        self.normm = nn.LayerNorm(m_dim)
    def forward(self,m,z,x,first):
        cb = x[:,-1]
        dismap=(cb[:,None,:]-cb[None,:,:]).norm(dim=-1)
        dis_z = fourier_encode_dist(dismap,self.dis_encoding_dim)
        if first:
            return 0,self.linear(dis_z)   
        else:
            z = self.normz(z) + self.linear(dis_z)   
            m = self.normm(m)
            return m,z 