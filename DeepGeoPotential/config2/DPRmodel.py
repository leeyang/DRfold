import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import basic,Evoformer,Embedder
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

class DPR(nn.Module):
    def __init__(self,msa_dim,ss_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim,n_head,c,n_layer,docheck,block_per_check):
        super(DPR,self).__init__()
        self.msa_dim = msa_dim
        self.N_ensemble=N_ensemble
        self.m_dim = m_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.loss_class = DPLoss()
        self.ppdim = self.loss_class.pp_bin+2
        self.ccdim = self.loss_class.cc_bin+2
        self.nndim = self.loss_class.nn_bin+2
        self.dihdim= self.loss_class.dih_bin + 1
        self.angdim= self.loss_class.ang_bin + 1
        self.n_head=n_head
        self.n_layer=n_layer
        self.c = c
        self.docheck=docheck
        self.block_per_check = block_per_check
        self.one_model = DPRIteration(msa_dim,ss_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim,n_head,c,n_layer,docheck,block_per_check)
        #self.structurenet = StructureModule.StructureModule(s_dim,z_dim,stru_N_layer=8,stru_N_head=16,stru_c=32)
        self.msa_predictor = basic.Linear(m_dim,msa_dim-1)

        self.pp_predictor = Task_module(z_dim,self.ppdim,True)
        self.cc_predictor = Task_module(z_dim,self.ccdim,True)
        self.nn_predictor = Task_module(z_dim,self.nndim,True)

        self.pcc_predictor = Task_module(z_dim,self.angdim,False)
        self.pnn_predictor = Task_module(z_dim,self.angdim,False)
        self.cnn_predictor = Task_module(z_dim,self.angdim,False)

        self.pccp_predictor = Task_module(z_dim,self.dihdim,True)
        self.pnnp_predictor = Task_module(z_dim,self.dihdim,True)
        self.cnnc_predictor = Task_module(z_dim,self.dihdim,True)



    def infer(self,x_dict):
        # msa: N_ilter, 1, L, d
        # ss : N_ilter, L, L, d
        # x_pre: L 3 3
        s_pre,z_pre=0,0
        N_ilter = x_dict['msa_mask'].shape[0]
        
        x_pre   = x_dict['pre_x']
        for i in range(1,N_ilter):
            
            s2,z,s,_,_ = self.one_model(x_dict,s_pre,z_pre,x_pre,i)
            #x = self.structurenet(s,z,x_dict,computeloss=False)['predx']
            s_pre = s2.detach()
            z_pre = z.detach()
            #x_pre = x.detach()
        s2,z,s,m,predss=self.one_model(x_dict,s_pre,z_pre,x_pre,N_ilter)

        pred_msa = F.log_softmax(self.msa_predictor(m),dim=-1)
        re_dict={}
        re_dict['pp'] = torch.exp(  self.pp_predictor(z) )
        re_dict['cc'] = torch.exp( self.cc_predictor(z))
        re_dict['nn'] = torch.exp( self.nn_predictor(z))

        re_dict['pcc'] = torch.exp( self.pcc_predictor(z))
        re_dict['pnn'] = torch.exp( self.pnn_predictor(z))
        re_dict['cnn'] = torch.exp( self.cnn_predictor(z)   )  

        re_dict['pccp'] = torch.exp( self.pccp_predictor(z))
        re_dict['pnnp'] = torch.exp( self.pnnp_predictor(z))
        re_dict['cnnc'] = torch.exp( self.cnnc_predictor(z)    )  

        return re_dict





class DPRIteration(nn.Module):
    def __init__(self,msa_dim,ss_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim,n_head,c,n_layer,docheck,block_per_check):
        super(DPRIteration,self).__init__()
        self.msa_dim = msa_dim
        self.N_ensemble=N_ensemble
        self.m_dim = m_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.n_head=n_head
        self.n_layer=n_layer
         
        self.c = c
        self.docheck=docheck
        self.block_per_check = block_per_check  
        self.msaembder = Embedder.MSAEncoder(msa_dim,m_dim,z_dim)
        self.ssembedder= Embedder.SSEncoder(ss_dim,z_dim)  
        self.re_emb=Embedder.RecyclingEmbedder(m_dim,z_dim,dis_encoding_dim=64)

        self.evmodel1 = Evoformer.Evoformer(m_dim,z_dim,n_head,c,n_layer[0],docheck,block_per_check)
        self.ss_predictor=basic.Linear(z_dim,1)
        self.ss_encoder  =basic.Linear(1,z_dim)
        self.evmodel2 = Evoformer.Evoformer(m_dim,z_dim,n_head,c,n_layer[1],docheck,block_per_check)
        self.slinear = basic.Linear(m_dim,s_dim)

    def forward(self,x_dict,s_pre,z_pre,x_pre,cycle_index):
        # x_dict['msa']
        msa_mask = (x_dict['msa_mask'][cycle_index-1])[:,:,None]
        ss_mask  = (x_dict['msa_mask'][cycle_index-1])[0]
        ss_mask  = 1-( (1-ss_mask)[:,None]*(1-ss_mask)[None,:]   )
        ss_mask  = ss_mask[:,:,None]
        msa_ = torch.cat([x_dict['msa']*(1-msa_mask),msa_mask],dim=-1) 
        ss_  = x_dict['ss'] + 0.0
        #print('maskmean',msa_mask.mean(),ss_mask.mean())
        ss_  = torch.cat([ss_*(1-ss_mask),ss_mask],dim=-1)
        N,L,_ = msa_.shape
        m,z = self.msaembder(msa_)
        z   = z + self.ssembedder(ss_)
        s2,z2=self.re_emb(s_pre,z_pre,x_pre,cycle_index==1)
        z   = z + z2
        m=torch.cat([(m[0]+s2)[None,...],m[1:]],dim=0)
        m,z=self.evmodel1(m,z)
        predss = torch.sigmoid(self.ss_predictor(z))
        z = z + self.ss_encoder(predss)
        m,z = self.evmodel2(m,z)


        s = self.slinear(m[0])
        return s2+m[0],z,s,m,predss




