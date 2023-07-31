import torch
from torch import nn
from torch.nn import functional as F
import basic,IPA



class TransitionModule(nn.Module):
    def __init__(self,c):
        super(TransitionModule,self).__init__()
        self.c=c
        self.norm1=nn.LayerNorm(c)
        self.linear1=basic.Linear(c,c)
        self.linear2=basic.Linear(c,c)
        self.linear3=basic.Linear(c,c)
        self.norm2=nn.LayerNorm(c)
    def forward(self,s_):
        s = self.norm1(s_)
        s = F.relu(   self.linear1(s) )
        s = F.relu(   self.linear2(s) )
        s = s_ + self.linear3(s)
        return self.norm2(s)

class BackboneUpdate(nn.Module):
    def __init__(self,indim):
        super(BackboneUpdate,self).__init__()
        self.indim=indim
        self.linear=basic.Linear(indim,6)
        torch.nn.init.zeros_(self.linear.linear.weight)
        torch.nn.init.zeros_(self.linear.linear.bias)
    def forward(self,s,L):
        pred=self.linear(s)
        rot=basic.quat2rot(pred[...,:3],L)
        return rot,pred[...,3:] #rot, translation

class TorsionNet(nn.Module):
    def __init__(self,s_dim,c):
        super(TorsionNet,self).__init__()
        self.s_dim=s_dim
        self.c=c
        self.linear1=basic.Linear(s_dim,c)
        self.linear2=basic.Linear(c,c)

        self.linear3=basic.Linear(c,c)
        self.linear4=basic.Linear(c,c) 

        self.linear5=basic.Linear(c,c)
        self.linear6=basic.Linear(c,c)

        self.linear7_1=basic.Linear(c,1)
        self.linear7_2=basic.Linear(c,2)
        self.linear7_3=basic.Linear(c,2)
    
    def forward(self,s_init,s):
        a = self.linear1(s_init) + self.linear2(s)
        a = a + self.linear4(F.relu(self.linear3(F.relu(a))))
        a = a + self.linear6(F.relu(self.linear5(F.relu(a))))
        bondlength = self.linear7_1(F.relu(a))
        angle = self.linear7_2(F.relu(a))
        torsion = self.linear7_3(F.relu(a))

        angle_L=torch.norm(angle,dim=-1,keepdim=True)
        angle = angle / (angle_L+1e-8)

        torsion_L = torch.norm(torsion,dim=-1,keepdim=True)
        torsion = torsion / (torsion_L+1e-8)

        return bondlength,angle,angle_L,torsion,torsion_L





    

class StructureModule(nn.Module):
    def __init__(self,s_dim,z_dim,N_layer,c):
        super(StructureModule,self).__init__()
        self.s_dim=s_dim
        self.z_dim=z_dim
        self.N_layer=N_layer
        self.N_head=8
        self.c=c
        self.use_rmsdloss=False
        self.layernorm_s=nn.LayerNorm(s_dim)
        self.layernorm_z=nn.LayerNorm(z_dim)
        #self.baseframe=self._base_frame()
        # shared weights part
        self.ipa=IPA.InvariantPointAttention(c,z_dim,c)
        self.transition = TransitionModule(c)
        self.bbupdate = BackboneUpdate(c)
        self.torsionnet=TorsionNet(s_dim,c)
        self._init_T()

    def _init_T(self):
        self.trans = torch.zeros(3)[None,:]
        self.rot = torch.eye(3)[None,:,:]


    def pred(self,s_init,z,base_x):
        if self.trans.device != s_init.device:
            self.trans=self.trans.to(s_init.device)
        if self.rot.device != s_init.device:
            self.rot=self.rot.to(s_init.device)
        L=s_init.shape[0]
        rot,trans=self.rot.repeat(L,1,1),self.trans.repeat(L,1)
        s = self.layernorm_s(s_init)
        z = self.layernorm_z(z)
        for layer in range(self.N_layer):
            s = s+ self.ipa(s,z,rot,trans)
            s = self.transition(s)
            rot_tmp,trans_tmp = self.bbupdate(s,L)
            rot,trans = basic.update_transform(rot_tmp,trans_tmp,rot,trans)
            
        s = s+ self.ipa(s,z,rot,trans)
        s = self.transition(s) 
        rot_tmp,trans_tmp = self.bbupdate(s,L)
        rot,trans = basic.update_transform(rot_tmp,trans_tmp,rot,trans)

        predx=base_x + 0.0
        predx = basic.batch_atom_transform(predx,rot,trans)
        return predx,rot,trans









    
        







        

