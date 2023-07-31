import torch
from torch import nn
from torch.nn import functional as F
import basic
import math


class TriOut(nn.Module):
    def __init__(self,z_dim,c):
        super(TriOut,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)
        self.inferbatch = 8
    def forward(self,z):
        z = self.norm(z)
        L = z.shape[1]
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        if self.training:
            o = torch.einsum('ilc,jlc->ijc',a,b)
        else:
            num_time = L//self.inferbatch
            if L%self.inferbatch!=0:
                num_time+=1
            o = [torch.einsum('ilc,jlc->ijc',a[:,i*self.inferbatch:(i+1)*self.inferbatch],b[:,i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.stack(o,dim=0).sum(dim=0)
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o

class TriIn(nn.Module):
    def __init__(self,z_dim,c):
        super(TriIn,self).__init__()
        self.z_dim = z_dim
        self.norm =nn.LayerNorm(z_dim)
        self.onorm =nn.LayerNorm(c)
        self.alinear=basic.Linear(z_dim,c)
        self.blinear=basic.Linear(z_dim,c)
        self.aglinear=basic.Linear(z_dim,c)
        self.bglinear=basic.Linear(z_dim,c)
        self.glinear =basic.Linear(z_dim,z_dim)
        self.olinear=basic.Linear(c,z_dim)
        self.inferbatch = 8
    def forward(self,z):
        L = z.shape[0]
        z = self.norm(z)
        a = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        b = self.alinear(z) * torch.sigmoid(self.aglinear(z))
        if self.training:
            o = torch.einsum('lic,ljc->ijc',a,b)
        else:
            num_time = L//self.inferbatch
            if L%self.inferbatch!=0:
                num_time+=1
            o = [torch.einsum('lic,ljc->ijc',a[i*self.inferbatch:(i+1)*self.inferbatch],b[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.stack(o,dim=0).sum(dim=0)           
        o = self.onorm(o)
        o = self.olinear(o)
        o = o * torch.sigmoid(self.glinear(z))
        return o


class TriAttStart(nn.Module):
    def __init__(self,z_dim,N_head,c):
        super(TriAttStart,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)
        self.inferbatch = 8
    def forward(self,z_):
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        if self.training:
            att = torch.einsum('blhc,bkhc->lkh',q,k)*(self.sq_c/math.sqrt(L2)) + b
            att = F.softmax(att,dim=1)
            o = torch.einsum('lkh,bkhc->blhc',att,v)
        else:
            num_time = L1//self.inferbatch
            if L1%self.inferbatch!=0:
                num_time+=1
            att = [torch.einsum('blhc,bkhc->lkh',q[i*self.inferbatch:(i+1)*self.inferbatch],k[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            att = (torch.stack(att,dim=0).sum(dim=0))  *(self.sq_c/math.sqrt(L2)) + b
            att = F.softmax(att,dim=1)

            o = [torch.einsum('lkh,bkhc->blhc',att,v[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.cat(o,dim=0)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        return o


class TriAttEnd(nn.Module):
    def __init__(self,z_dim,N_head=12,c=24):
        super(TriAttEnd,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)
        self.inferbatch = 8
    def forward(self,z):
        z_ = z.permute(1,0,2)
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z).permute(1,0,2)
        if self.training:
            att = torch.einsum('blhc,bkhc->lkh',q,k)*(self.sq_c/math.sqrt(L2)) + b
            att = F.softmax(att,dim=1)
            o = torch.einsum('lkh,bkhc->blhc',att,v)
        else:
            num_time = L1//self.inferbatch
            if L1%self.inferbatch!=0:
                num_time+=1
            att = [torch.einsum('blhc,bkhc->lkh',q[i*self.inferbatch:(i+1)*self.inferbatch],k[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            att = (torch.stack(att,dim=0).sum(dim=0))  *(self.sq_c/math.sqrt(L2)) + b
            att = F.softmax(att,dim=1)

            o = [torch.einsum('lkh,bkhc->blhc',att,v[i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.cat(o,dim=0)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        return o.permute(1,0,2)
class TriAttEnd_old(nn.Module):
    def __init__(self,z_dim,N_head=12,c=24):
        super(TriAttEnd_old,self).__init__()
        self.z_dim = z_dim
        self.N_head = N_head
        self.c = c
        self.sq_c = 1/math.sqrt(c)
        self.norm=nn.LayerNorm(z_dim)
        self.qlinear=basic.Linear(z_dim,c*N_head)
        self.klinear=basic.Linear(z_dim,c*N_head)
        self.vlinear=basic.Linear(z_dim,c*N_head)
        self.blinear=basic.Linear(z_dim,N_head)
        self.glinear=basic.Linear(z_dim,c*N_head)
        self.olinear=basic.Linear(c*N_head,z_dim)
        self.inferbatch = 8
    def forward(self,z_):
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        if self.training:
            att = torch.einsum('blhc,kbhc->blkh',q,k)*self.sq_c + b[None,:,:,:].permute(0,2,1,3)
            att = F.softmax(att,dim=2)
            o = torch.einsum('blkh,klhc->blhc',att,v)
        else:
            
            num_time = L2//self.inferbatch
            if L2%self.inferbatch!=0:
                num_time+=1
            att = [torch.einsum('blhc,kbhc->blkh',q[i*self.inferbatch:(i+1)*self.inferbatch],k[:,i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            att = torch.cat(att,dim=0)  *self.sq_c + b[None,:,:,:].permute(0,2,1,3)
            att = F.softmax(att,dim=2)
            o = [torch.einsum('blkh,klhc->blhc',att[:,i*self.inferbatch:(i+1)*self.inferbatch],v[:,i*self.inferbatch:(i+1)*self.inferbatch]) for i in range(num_time)]
            o = torch.cat(o,dim=1)
            #o = torch.einsum('blkh,klhc->blhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        return o
    def forward2(self,z_):
        z = z_.permute(1,0,2)
        L1,L2,D=z_.shape
        z = self.norm(z_)
        q = self.qlinear(z).reshape(L1,L2,self.N_head,self.c)
        k = self.klinear(z).reshape(L1,L2,self.N_head,self.c)
        v = self.vlinear(z).reshape(L1,L2,self.N_head,self.c)
        b = self.blinear(z)
        att = torch.einsum('blhc,bkhc->blkh',q,k)*self.sq_c + b[None,:,:,:]
        att = F.softmax(att,dim=2)
        o = torch.einsum('blkh,bkhc->blhc',att,v)
        o = (torch.sigmoid(self.glinear(z).reshape(L1,L2,self.N_head,self.c)) * o).reshape(L1,L2,-1)
        o = self.olinear(o)
        o = o.permute(1,0,2)
        return o
class PairTrans(nn.Module):
    def __init__(self,z_dim,c_expand):
        super(PairTrans,self).__init__()
        self.z_dim=z_dim
        self.c_expand=c_expand
        self.norm = nn.LayerNorm(z_dim)
        self.linear1=basic.Linear(z_dim,z_dim*c_expand)
        self.linear2=basic.Linear(z_dim*c_expand,z_dim)
    def forward(self,z):
        a = self.linear1(self.norm(z))
        a = self.linear2(F.relu(a))
        return a 




