from numpy import select
import torch
from torch import nn
from torch.nn import functional as F
import basic
import math


class MSARow(nn.Module):
    def __init__(self,m_dim,z_dim,N_head,c):
        super(MSARow,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.klinear = basic.LinearNoBias(m_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.norm_z  = nn.LayerNorm(z_dim)
        self.zlinear = basic.LinearNoBias(z_dim,N_head)
        self.glinear = basic.Linear(m_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,m_dim)
        self.inferbatch = 8
    def forward(self,m,z):
        # m : N L 32  
        N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(N,L,self.N_head,self.c)
        b = self.zlinear(self.norm_z(z))
        g = torch.sigmoid(self.glinear(m)).reshape(N,L,self.N_head,self.c)
        # rq rv h
        if self.training:
            att=torch.einsum('bqhc,bvhc->qvh',q,k) * (self.sq_c/math.sqrt(N)) + b
            att=F.softmax(att,dim=1)
            o = torch.einsum('qvh,bvhc->bqhc',att,v) * g
        else:

            num_time = N//self.inferbatch
            if N%self.inferbatch!=0:
                num_time+=1
            att = [torch.einsum('bqhc,bvhc->qvh',q[i*self.inferbatch:(i+1)*self.inferbatch],k[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            att = (torch.stack(att,dim=0).sum(dim=0)) * (self.sq_c/math.sqrt(N)) + b
            att=F.softmax(att,dim=1)
            o = [torch.einsum('qvh,bvhc->bqhc',att,v[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time) ]
            o = torch.cat(o,dim=0) * g

        

        m_ = self.olinear(o.reshape(N,L,-1))
        return m_

class MSACol(nn.Module):
    def __init__(self,m_dim,N_head,c):
        super(MSACol,self).__init__()
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm1=nn.LayerNorm(m_dim)
        self.qlinear = basic.LinearNoBias(m_dim,N_head*c)
        self.klinear = basic.LinearNoBias(m_dim,N_head*c)
        self.vlinear = basic.LinearNoBias(m_dim,N_head*c)

        self.glinear = basic.Linear(m_dim,N_head*c)
        self.olinear = basic.Linear(N_head*c,m_dim)
        self.inferbatch = 8
    def forward(self,m):
        # m : N L 32  
        N,L,D = m.shape  
        m = self.norm1(m)
        q = self.qlinear(m).reshape(N,L,self.N_head,self.c) #s rq h c 
        k = self.klinear(m).reshape(N,L,self.N_head,self.c) #s rv h c 
        v = self.vlinear(m).reshape(N,L,self.N_head,self.c)

        g = torch.sigmoid(self.glinear(m)).reshape(N,L,self.N_head,self.c)
        if self.training:
            att=torch.einsum('slhc,tlhc->sth',q,k) * (self.sq_c/math.sqrt(L))  # rq rv h
            att=F.softmax(att,dim=1)
            o = torch.einsum('sth,tlhc->slhc',att,v) * g
        else:
            num_time = L//self.inferbatch
            if L%self.inferbatch!=0:
                num_time+=1  
            att = [torch.einsum('slhc,tlhc->sth',q[:,i*self.inferbatch:(i+1)*self.inferbatch],k[:,i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            att = (torch.stack(att,dim=0).sum(dim=0)) * (self.sq_c/math.sqrt(L))
            att=F.softmax(att,dim=1)   
            o = [torch.einsum('sth,tlhc->slhc',att,v[:,i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time) ]
            o = torch.cat(o,dim=1) * g      
        m_ = self.olinear(o.reshape(N,L,-1))
        return m_

class MSATrans(nn.Module):
    def __init__(self,m_dim):
        super(MSATrans,self).__init__()
        self.c_expand=2
        c_expand=2
        self.m_dim=m_dim
        self.norm=nn.LayerNorm(m_dim)
        self.linear1 = basic.Linear(m_dim,m_dim*c_expand)
        self.linear2 = basic.Linear(m_dim*c_expand,m_dim)
    def forward(self,m):
        m = self.norm(m)
        m = self.linear1(m)
        m = self.linear2(F.relu(m))
        return m

class MSAOPM(nn.Module):
    def __init__(self,m_dim,z_dim,c):
        super(MSAOPM,self).__init__()
        self.m_dim=m_dim
        self.c=c
        self.norm=nn.LayerNorm(m_dim)
        self.linear1=basic.Linear(m_dim,c)
        self.linear2=basic.Linear(m_dim,c)
        self.linear3=basic.Linear(c*c,z_dim)
        self.inferbatch = 8
    def forward(self,m):
        N,L,D=m.shape
        o=self.norm(m)
        a=self.linear2(o)
        b=self.linear1(o)
        if self.training:
            o = torch.einsum('nia,njb->nijab',a,b).mean(dim=0)
        else:
            num_time = N//self.inferbatch
            if N%self.inferbatch!=0:
                num_time+=1
            o = [torch.einsum('nia,njb->nijab',a[i*self.inferbatch:(i+1)*self.inferbatch],b[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.cat(o,dim=0).mean(dim=0)
        o = self.linear3(o.reshape(L,L,-1))
        return o


        



