import os,sys
import numpy as np 
from pathlib import Path
import shutil
##############please modify##################
petfoldbin =f'/home/liyangum/projects/RNA/DRfold/bin/PETfold/bin/PETfold'
ViennaRNAbin='/home/liyangum/projects/RNA/DRfold/bin/ViennaRNA-2.4.18-bin/bin/RNAfold'
#############################################


def ss2matrix(ssstr):
    L=len(ssstr)
    ssmatrix = np.zeros( [L,L])
    slist=[]
    for i in range(L):
        if ssstr[i] in ['(','<','[','{']:
            slist.append(i)
        elif ssstr[i] in [')','>',']','}']:
            j = slist[-1]
            ssmatrix[i,j]=1
            ssmatrix[j,i]=1
            slist.pop(-1)
        elif ssstr[i] not in  ['.','-']:
            print('unknown ss state',ssstr[i], i)
            assert False
    return ssmatrix
def PETfold_runner(fastafile,saveprefix):
    
    petfold_savefile=saveprefix+'.petfold'
    ppfile          =saveprefix+'.petfoldrr'
    fasta_ = os.path.abspath(fastafile)
    cmd = f'{petfoldbin} -f {fasta_} -r {ppfile} >{petfold_savefile}'
    print(cmd)
    os.system(cmd)
    petfoldline=open(petfold_savefile).readlines()[-2].strip()
    ss1=ss2matrix(petfoldline.split()[-1])
    ss2=np.genfromtxt(ppfile,skip_header=1,skip_footer=1)
    seq = open(fastafile).readlines()[1].strip()
    L = len(seq)
    L1,L2 = ss2.shape
    L3 = ss1.shape[0]
    os.remove(petfold_savefile)
    os.remove(ppfile)
    if L==L1 and L==L2 and L==L3:
        #np.save(saveprefix,np.stack(ss1,ss2))
        return np.concatenate([ss1[...,None],ss2[...,None]],axis=-1)
    else:
        print('PETfold lengths does not id',L,L1,L2,L3)
        assert False


def ViennaRNA_runner(fastafile,saveprefix_):
    
    saveprefix = os.path.abspath(saveprefix_)
    path = Path(saveprefix)
    seq=open(fastafile).readlines()[1].strip()
    workingdir = path.parent.absolute()
    name = os.path.basename(saveprefix)
    newfastafile = saveprefix+'.tmp.fasta'
    lines=[f'>{name}_ViennaRNA_name',seq]
    wfile =open(newfastafile,'w')
    wfile.write('\n'.join(lines))
    wfile.close()
    os.chdir(workingdir)
    cmd = f'{ViennaRNAbin} --outfile={name}.ViennaRNAss -p  --noPS {os.path.abspath(newfastafile)}'
    print(cmd)
    os.system(cmd)
    ss_str = open(f'{saveprefix}.ViennaRNAss').readlines()[2].strip().split()[0]
    ss1=ss2matrix(ss_str)
    L=len(seq)
    pslines=open(os.path.join(workingdir,f'{name}_ViennaRNA_name_dp.ps')).readlines()
    pslines=[aline.strip() for aline in pslines]
    pslines=[aline.strip() for aline in pslines if aline.endswith('ubox') and len(aline.split())==4 and aline[0].isdigit()]

    ss2 = np.zeros([L,L])
    for aline in pslines:
        words = aline.split()
        i,j,score = int(words[0])-1,int(words[1])-1,float(words[2])
        ss2[i,j]=score
        ss2[j,i]=score
    os.remove(os.path.join(workingdir,f'{name}_ViennaRNA_name_dp.ps'))
    os.remove(newfastafile)
    os.remove(f'{saveprefix}.ViennaRNAss')
    if L==ss1.shape[0]:
        return np.concatenate([ss1[...,None],ss2[...,None]],axis=-1)
    else:
        print('ViennaRNA lengths does not id',L,ss1.sape)
        assert False


def Extraxt_ss(fastafile,saveprefix):
    savepath = os.path.abspath(saveprefix) +''
    ss1 = PETfold_runner(fastafile,saveprefix+'_pet')
    ss2 = ViennaRNA_runner(fastafile,saveprefix+'_vie')
    ss = np.concatenate([ss1,ss2],axis=-1)
    np.save(savepath,ss)
    return ss


def mutN(fastafile,saveprefix,outfasta):
    # A U; GC
    conj={'A':'U','U':'A','G':'C','C':'G'}
    seq = open(fastafile).readlines()[1].strip()
    seq = list(seq)
    ss = np.load(saveprefix+'.npy')[:,:,0]
    newseq=[]
    for i in range(len(seq)):
        if seq[i] not in ['A','G','C','U']:
            contacts = ss[i] # L
            if contacts.sum()<0.5:# nno ss
                newseq.append('U')
            else:
                contacts2 = list(contacts.astype(int))
                if 1 in contacts2:
                    j = contacts2.index(1)
                    if seq[j] in ['A','U','G','C']:
                        newseq.append(conj[seq[j]])
                    else:
                        newseq.append('U')
                else:
                    newseq.append('U')
        else:
            newseq.append(seq[i])
    wfile = open(outfasta,'w')
    wfile.write('>test\n')
    wfile.write(''.join(newseq))
    wfile.close()



    
    

if __name__ == '__main__':
    fastafile=sys.argv[1]
    saveprefix='/nfs/amino-home/liyangum/eRNA/programs/example/test_ly'
    saveprefix=sys.argv[2]
    outfasta = sys.argv[3]
    Extraxt_ss(fastafile,saveprefix)
    mutN(fastafile,saveprefix,outfasta)