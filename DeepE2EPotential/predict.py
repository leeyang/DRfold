import os,sys,random
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.nn import functional as F

import EvoMSA2XYZ

#config

msa_dim=6+1
m_dim,s_dim,z_dim = 64,64,64
N_ensemble,N_cycle=3,4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model=EvoMSA2XYZ.MSA2XYZ(msa_dim-1,msa_dim,N_ensemble,N_cycle,m_dim,s_dim,z_dim)
    model_dict=torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)
    model.eval()
    return model
def parse_seq(inseq):
    seqnpy=np.zeros(len(inseq))
    seq1=np.array(list(inseq))
    seqnpy[seq1=='A']=1
    seqnpy[seq1=='G']=2
    seqnpy[seq1=='C']=3
    seqnpy[seq1=='U']=4
    seqnpy[seq1=='T']=4
    return seqnpy


def Get_base(seq,basenpy_standard):
    basenpy = np.zeros([len(seq),3,3])
    seqnpy = np.array(list(seq))
    basenpy[seqnpy=='A']=basenpy_standard[0]
    basenpy[seqnpy=='a']=basenpy_standard[0]

    basenpy[seqnpy=='G']=basenpy_standard[1]
    basenpy[seqnpy=='g']=basenpy_standard[1]

    basenpy[seqnpy=='C']=basenpy_standard[2]
    basenpy[seqnpy=='c']=basenpy_standard[2]

    basenpy[seqnpy=='U']=basenpy_standard[3]
    basenpy[seqnpy=='u']=basenpy_standard[3]

    basenpy[seqnpy=='T']=basenpy_standard[3]
    basenpy[seqnpy=='t']=basenpy_standard[3]
    return basenpy
def todevice(adict,device):
    for aterm in adict.keys():
        adict[aterm]=torch.FloatTensor(adict[aterm]).to(device)
    return adict
def tonumpy(adict):
    for aterm in adict.keys():
        adict[aterm]=adict[aterm].cpu().data. numpy()
    return adict




def pipeline(fastafile,ssfile,savefile_):
    target = {}
    seq=open(fastafile).readlines()[1].strip()
    d=len(seq)
    basenpy = np.load( os.path.join(os.path.dirname(os.path.abspath(__file__)),'base.npy'  )  )
    ss =np.load(ssfile)

    msa=torch.from_numpy(parse_seq(seq))[None,:]
    msa=torch.cat([msa,msa],0)
    msa=F.one_hot(msa.long(),6).float().to(device)
    ss = torch.FloatTensor(ss).to(device)
    base_x = torch.FloatTensor(Get_base(seq,basenpy) ).to(device)

    model_list = os.listdir(  os.path.join(os.path.dirname(__file__),'model')   )
    model_list =[amodel for amodel in model_list if amodel.startswith('model')]
    count=0
    for amodel in model_list:
        #amodel = model_list[0]
        model_path=os.path.join(os.path.dirname(__file__),'model',amodel)
        model=load_model(model_path).to(device)
        with torch.no_grad():
            N_cycle = 4
            predxs = model.pred(msa,ss,base_x,N_cycle)
            coor = predxs[N_cycle-1].numpy()
            np.save(savefile_+'_'+str(count),coor)
            count+=1

if __name__ == '__main__':
    seqfile=sys.argv[1]
    ssfile = sys.argv[2]
    savefile=sys.argv[3]
    pipeline(seqfile,ssfile,os.path.abspath(savefile))
    