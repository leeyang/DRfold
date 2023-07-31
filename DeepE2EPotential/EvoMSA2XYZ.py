import torch
from torch import nn
from torch.nn import functional as F
import basic,Structure,Evoformer
import numpy as np







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
        self.pre_z=nn.Linear(4,z_dim)
        self.premsa=PreMSA(seq_dim,msa_dim,m_dim,z_dim)
        self.re_emb=RecyclingEmbedder(m_dim,z_dim,dis_encoding_dim=64)
        self.evmodel=Evoformer.Evoformer(m_dim,z_dim,True)    
        self.slinear=basic.Linear(z_dim,s_dim)

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
        
        self.structurenet=Structure.StructureModule(s_dim,z_dim,4,s_dim) #s_dim,z_dim,N_layer,c)


    def pred(self,msa_,ss,base_x,n_cycle):
        predxs={}
        scores={}
        L=msa_.shape[1]
        m1_pre,z_pre=0,0
        x_pre=torch.zeros(L,3,3).to(msa_.device)
        for i in range(n_cycle):
            m1,z,s=self.msaxyzone.pred(msa_,ss,m1_pre,z_pre,x_pre,i)
            x = self.structurenet.pred(s,z,base_x)[0]
            m1_pre=m1.detach()
            z_pre = z.detach()
            x_pre = x.detach()
            pred_dis = F.softmax(self.dis_predor(z),dim=-1) 
            score =0
            predxs[i]=x_pre.cpu().detach()
            predxs[str(i)+'_score']=score

        ########################last cycle###########

        return predxs   

