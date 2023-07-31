from numpy import select
import torch
from torch import nn
from torch.nn import functional as F
import basic,Evoformer,EvoPair,EvoMSA
import math
from torch.utils.checkpoint import checkpoint
import numpy as np


class DPLoss:
    def __init__(self) -> None:
        self.pp_th = 30
        self.pp_bin= 56

        self.cc_th = 24
        self.cc_bin= 44

        self.nn_th = 18
        self.nn_bin= 32

        self.dih_bin=36
        self.ang_bin=18


class Task_module(nn.Module):
    def __init__(self,indim,odim,symm):
        super(Task_module,self).__init__()
        self.c=c=indim
        self.norm1=nn.LayerNorm(c)
        self.linear1=basic.Linear(c,c)
        self.linear2=basic.Linear(c,c)
        self.linear3=basic.Linear(c,c)
        self.norm2=nn.LayerNorm(c)
        self.linear4 = basic.Linear(c,odim)
        self.symm = symm
    def forward(self,s_):
        s = self.norm1(s_)
        s = F.relu(   self.linear1(s) )
        s = F.relu(   self.linear2(s) )
        s = s_ + self.linear3(s)
        s = self.linear4(  self.norm2(s))
        if self.symm:
            s = (s + s.permute(1,0,2))*0.5
        return F.log_softmax(s,dim=-1)


class PreMSA(nn.Module):
    def __init__(self,seq_dim,msa_dim,m_dim,z_dim):
        super(PreMSA,self).__init__()
        self.msalinear=basic.Linear(msa_dim,m_dim)
        self.qlinear  =basic.Linear(seq_dim,z_dim)
        self.klinear  =basic.Linear(seq_dim,z_dim)
        self.slinear  =basic.Linear(seq_dim,m_dim)
        self.pos = self.compute_pos().float()
        self.pos1d=self.compute_apos()
        self.poslinear=basic.Linear(65,z_dim)
        self.poslinear2=basic.Linear(14,m_dim)
    def tocuda(self,device):
        self.to(device)
        self.pos.to(device)
    def compute_apos(self,maxL=2000):
        d = torch.arange(maxL)
        m = 14
        d =(((d[:,None] & (1 << np.arange(m)))) > 0).float()
        return d

    def compute_pos(self,maxL=2000):
        a = torch.arange(maxL)
        b = (a[None,:]-a[:,None]).clamp(-32,32)
        return F.one_hot(b+32,65)


    def forward(self,seq,msa):
        if self.pos.device != msa.device:
            self.pos = self.pos.to(msa.device)
        if self.pos1d.device != msa.device:
            self.pos1d = self.pos1d.to(msa.device)
        # msa N L D, seq L D
        N,L,D=msa.shape
        s = self.slinear(seq)
        m = self.msalinear(msa)
        p = self.poslinear2(self.pos1d[:L])

        m = m + s[None,:,:] + p[None,:,:]
        

        sq=self.qlinear(seq)
        sk=self.klinear(seq)
        z=sq[None,:,:]+sk[:,None,:]

        z = z + self.poslinear( self.pos[:L,:L]  )
        return m,z

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

class MSA2xyzIteration(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2xyzIteration,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.seq_dim=seq_dim
        self.N_ensemble=N_ensemble
        self.dis_dim=36 + 2 
        self.pre_z=nn.Linear(1,z_dim)
        self.premsa=PreMSA(seq_dim,msa_dim,m_dim,z_dim)
        self.re_emb=RecyclingEmbedder(m_dim,z_dim,dis_encoding_dim=64)
        self.evmodel=Evoformer.Evoformer(m_dim,z_dim,True)    
        self.slinear=basic.Linear(z_dim,s_dim)

    def forward(self,msa_,ss_,m1_pre,z_pre,pre_x,cycle_index):

        m1_all,z_all,s_all=0,0,0
        N,L,_=msa_.shape
        
        for i in range(1):
            ##################random sample##################################
            msa_mask = torch.zeros(N,L).to(msa_.device)
            msa_true = msa_ + 0
            #msa_mask=basic.Generate_msa_mask(num_tosam+1,L).to(msa_.device) # +1 means plus seq
            ####################################################################
            seq = msa_true[0]*1.0 # 22-dim
            msa = torch.cat([msa_true*(1-msa_mask[:,:,None]),msa_mask[:,:,None]],dim=-1)
            m,z=self.premsa(seq,msa)
            ss = self.pre_z(ss_)
            z  = z+ss
            #m,z=checkpoint(self.premsa,seq,msa)
            m1_,z_=self.re_emb(m1_pre,z_pre,pre_x,cycle_index==0) #already added residually
            #m1_,z_=checkpoint(self.re_emb,m1_pre,z_pre,pre_x,cycle_index==0)
            z = z+z_
            m=torch.cat([(m[0]+m1_)[None,...],m[1:]],dim=0)
            m,z=self.evmodel(m,z)
            s = self.slinear(m[0])
            m1_all =m1_all + m[0]
            z_all  =z_all  + z
            s_all  =s_all + s
        return m1_all,z_all,s_all,m,msa_true,msa_mask
    def pred(self,msa_,ss_,m1_pre,z_pre,pre_x,cycle_index):
        m1_all,z_all,s_all=0,0,0
        N,L,_=msa_.shape
        for i in range(self.N_ensemble):
            msa_mask = torch.zeros(N,L).to(msa_.device)
            msa_true = msa_ + 0
            seq = msa_true[0]*1.0 # 22-dim
            msa = torch.cat([msa_true*(1-msa_mask[:,:,None]),msa_mask[:,:,None]],dim=-1)
            m,z=self.premsa(seq,msa)
            ss = self.pre_z(ss_)
            z  = z+ss
            m1_,z_=self.re_emb(m1_pre,z_pre,pre_x,cycle_index==0) #already added residually
            z = z+z_
            m=torch.cat([(m[0]+m1_)[None,...],m[1:]],dim=0)
            m,z=self.evmodel(m,z)
            s = self.slinear(m[0])
            m1_all =m1_all + m[0]
            z_all  =z_all  + z
            s_all  =s_all + s
        return m1_all/self.N_ensemble,z_all/self.N_ensemble,s_all/self.N_ensemble


class MSA2XYZ(nn.Module):
    def __init__(self,seq_dim,msa_dim,N_ensemble,N_cycle,m_dim=64,s_dim=128,z_dim=64,docheck=True):
        super(MSA2XYZ,self).__init__()
        self.msa_dim=msa_dim
        self.m_dim=m_dim
        self.z_dim=z_dim
        self.dis_dim=36 + 2
        self.N_cycle=N_cycle
        self.msaxyzone = MSA2xyzIteration(seq_dim,msa_dim,N_ensemble,m_dim=m_dim,s_dim=s_dim,z_dim=z_dim)
        self.msa_predor=basic.Linear(m_dim,msa_dim-1)
        self.dis_predor=basic.Linear(z_dim,self.dis_dim)
        self.slinear=basic.Linear(m_dim,s_dim)
        self.loss_class = DPLoss()
        self.ppdim = self.loss_class.pp_bin+2
        self.ccdim = self.loss_class.cc_bin+2
        self.nndim = self.loss_class.nn_bin+2
        self.dihdim= self.loss_class.dih_bin + 1
        self.angdim= self.loss_class.ang_bin + 1    
        self.pp_predictor = Task_module(z_dim,self.ppdim,True)
        self.cc_predictor = Task_module(z_dim,self.ccdim,True)
        self.nn_predictor = Task_module(z_dim,self.nndim,True)

        self.pcc_predictor = Task_module(z_dim,self.angdim,False)
        self.pnn_predictor = Task_module(z_dim,self.angdim,False)
        self.cnn_predictor = Task_module(z_dim,self.angdim,False)

        self.pccp_predictor = Task_module(z_dim,self.dihdim,True)
        self.pnnp_predictor = Task_module(z_dim,self.dihdim,True)
        self.cnnc_predictor = Task_module(z_dim,self.dihdim,True)
    


    def pred(self,msa_,ss):
        predxs={}
        L=msa_.shape[1]
        m1_pre,z_pre=0,0
        x_pre=torch.zeros(L,3,3).to(msa_.device)
        for i in range(self.N_cycle):
            m1,z,s=self.msaxyzone.pred(msa_,ss,m1_pre,z_pre,x_pre,i)
            #x = self.structurenet.pred(s,z)[0]
            m1_pre=m1.detach()
            z_pre = z.detach()
            #x_pre = x.detach()
            predxs[i]=x_pre.cpu().detach()
            re_dict={}
            re_dict['pp'] = self.pp_predictor(z)
            re_dict['cc'] = self.cc_predictor(z)
            re_dict['nn'] = self.nn_predictor(z)

            re_dict['pcc'] = self.pcc_predictor(z)
            re_dict['pnn'] = self.pnn_predictor(z)
            re_dict['cnn'] = self.cnn_predictor(z)     

            re_dict['pccp'] = self.pccp_predictor(z)
            re_dict['pnnp'] = self.pnnp_predictor(z)
            re_dict['cnnc'] = self.cnnc_predictor(z)    
        ########################last cycle###########

        return predxs,re_dict   



