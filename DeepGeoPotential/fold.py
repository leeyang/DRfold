#!/nfs/amino-home/liyangum/miniconda3/bin/python
import os,sys

import numpy as np
from pyrosetta import (
    rosetta,
    MoveMap,
    SwitchResidueTypeSetMover,
    ScoreFunction,
    init,
    create_score_function,
    RepeatMover,
    pose_from_sequence,
)
from pyrosetta.rosetta.protocols.minimization_packing import MinMover



def init_structure(seq_):
    seq=seq_.lower()
    seq = seq.replace('t','u')
    print('seq:',seq)
    assembler = rosetta.core.import_pose.RNA_HelixAssembler()
    assembled_pose = assembler.build_init_pose(seq, '')
    return assembled_pose

def get_AtomID(pose, resi, atomname) :
    residue = pose.residue(resi)
    return rosetta.core.id.AtomID(atomno_in=residue.atom_index(atomname), rsd_in=resi)

def gen_rst(npy,seq_,pos,std=1, tol=0.5):
    seq = seq_.lower()
    seq = seq.replace('t','u')
    L = npy.shape[0]
    rst =[]
    flat_har = rosetta.core.scoring.func.HarmonicFunc(0.0, std)
    fixed_pt = pos.atom_tree().root().atom_id()
    #flat_har = rosetta.core.scoring.func.FlatHarmonicFunc(0.0, std, tol)
    for i in range(L):
        res_id = i+1
        if seq[i] in ['a','g']:
            atoms = ['P','C4','N9']
            #atoms = ['P','C4']

        elif seq[i] in ['c','u']:
            atoms = ['P','C4','N1']
            #atoms = ['P','C4']

        else:
            assert seq[i] in  ['a','g'] + ['c','u']
        for j,atm in zip(range(len(atoms)),atoms):
            xyz=rosetta.numeric.xyzVector_double_t(npy[i,j,0],npy[i,j,1],npy[i,j,2])
            ida = get_AtomID(pos,res_id,atm)
            rst.append(rosetta.core.scoring.constraints.CoordinateConstraint(ida, ida, xyz, flat_har))
    return rst
    constraints = rosetta.core.scoring.constraints.ConstraintSet()
    for constraint in rst:
        constraints.add_constraint(constraint)

    # add to pose
    csm = rosetta.protocols.constraint_movers.ConstraintSetMover()
    csm.constraint_set(constraints)
    csm.add_constraints(True)
    csm.apply(pos)

def gen_dist_rst(npy,seq_,pos,confidence_prop,sep,min_dis,max_dis,num_bin,atom_type_):
    assert atom_type_ in ['P','C','N']

    rst=[]

    seq = seq_.lower()
    seq = seq.replace('t','u')
    L,_,n_bin = npy.shape
    prob =  1- npy[:,:,-1]
    potentials = -np.log(npy[:,:,1:-1]+1e-4)
    potentials = potentials - potentials[:,:,[-1]]
    potentials = np.concatenate([potentials[:,:,[0]],potentials],axis=-1)
    step_size = (max_dis-min_dis)*1.0/num_bin
    bins = [0] + [min_dis + (i+0.5)*step_size for i in range(num_bin)]
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    i, j = np.where(prob > confidence_prop)
    prob = prob[i, j]
    for a, b, p in zip(i, j, prob):
        if b > a and abs(a - b) >= sep :
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in potentials[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, step_size, rosetta_spline_bins, rosetta_spline_potential
            )
            if atom_type_ =='P':
                atom_type1= atom_type2= 'P'
            elif atom_type_=='C':
                atom_type1 =atom_type2= 'C4'
            elif atom_type_=='N':
                if seq[a] in ['a','g']:
                    atom_type1 = 'N9'
                else:
                    atom_type1= 'N1'
                if seq[b] in ['a','g']:
                    atom_type2 = 'N9'
                else:
                    atom_type2= 'N1'
            else:
                assert False
            atom_id_a = get_AtomID(pos,a+1,atom_type1)#rosetta.core.id.AtomID(5, a + 1)
            atom_id_b = get_AtomID(pos,b+1,atom_type2) # 5 is CB
            rst.append(
                rosetta.core.scoring.constraints.AtomPairConstraint(atom_id_a, atom_id_b, spline)
            )
    print(f" {atom_type_} dist restraints: {len(rst)}")
    return rst

def deter_atom(atom_type_,aa):
    if atom_type_ =='P':
        atom_type1= 'P'
    elif atom_type_=='C':
        atom_type1 = 'C4'
    elif atom_type_=='N':
        if aa in ['a','g']:
            atom_type1 = 'N9'
        else:
            atom_type1= 'N1'
    else:
        assert False    
    return atom_type1

def gen_dih_rst(npy,seq_,pos,confidence_prop,sep,atom_type_):
    for anatom in atom_type_:
        assert anatom in ['P','C','N']
    rst=[]

    seq = seq_.lower()
    seq = seq.replace('t','u')
    L,_,n_bin = npy.shape
    ASTEP = np.deg2rad(np.pi*2/(n_bin-1))
    n_bin = n_bin -1 + 4
    prob =  1- npy[:,:,-1]
    bins = np.linspace(-np.pi - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, n_bin)
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    potentials = -np.log(npy[:,:,:-1]+1e-4)
    potentials = np.concatenate([potentials[:, :, -2:], potentials, potentials[:, :, :2]], axis=-1)
    i, j = np.where(prob > confidence_prop)
    prob = prob[i, j]
    for a, b, p in zip(i, j, prob):
        if b > a and abs(a - b) >= sep :
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in potentials[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, ASTEP, rosetta_spline_bins, rosetta_spline_potential
            )  
            atom_id_1 = get_AtomID(pos,a+1,deter_atom(atom_type_[0],seq[a]))     
            atom_id_2 = get_AtomID(pos,a+1,deter_atom(atom_type_[1],seq[a]))    
            atom_id_3= get_AtomID(pos,b+1,deter_atom(atom_type_[2],seq[b]))  
            atom_id_4 = get_AtomID(pos,b+1,deter_atom(atom_type_[3],seq[b]))     
            rst.append(
                rosetta.core.scoring.constraints.DihedralConstraint(atom_id_1, atom_id_2, atom_id_3, atom_id_4, spline)
            )
    print(f" {atom_type_} dih restraints: {len(rst)}")
    return rst
def gen_ang_rst(npy,seq_,pos,confidence_prop,sep,atom_type_):
    for anatom in atom_type_:
        assert anatom in ['P','C','N']
    rst=[]

    seq = seq_.lower()
    seq = seq.replace('t','u')
    L,_,n_bin = npy.shape
    ASTEP = np.deg2rad(np.pi/(n_bin-1))
    n_bin = n_bin -1 + 4
    prob =  1- npy[:,:,-1]
    bins = np.linspace(0 - 1.5 * ASTEP, np.pi + 1.5 * ASTEP, n_bin)
    rosetta_spline_bins = rosetta.utility.vector1_double()
    for bin_ in bins:
        rosetta_spline_bins.append(bin_)
    potentials = -np.log(npy[:,:,:-1]+1e-4)
    potentials = np.concatenate([potentials[:, :, [0]],potentials[:, :, [0]], potentials, potentials[:, :, [-1]],potentials[:, :, [-1]]], axis=-1)
    i, j = np.where(prob > confidence_prop)
    prob = prob[i, j]
    for a, b, p in zip(i, j, prob):
        if abs(a - b) >= sep :
            rosetta_spline_potential = rosetta.utility.vector1_double()
            for pot in potentials[a, b]:
                rosetta_spline_potential.append(pot)
            spline = rosetta.core.scoring.func.SplineFunc(
                "", 1.0, 0.0, ASTEP, rosetta_spline_bins, rosetta_spline_potential
            )  
            atom_id_1 = get_AtomID(pos,a+1,deter_atom(atom_type_[0],seq[a]))     
            atom_id_2 = get_AtomID(pos,a+1,deter_atom(atom_type_[1],seq[a]))    
            atom_id_3= get_AtomID(pos,b+1,deter_atom(atom_type_[2],seq[b]))     
            rst.append(
                rosetta.core.scoring.constraints.AngleConstraint(atom_id_1, atom_id_2, atom_id_3, spline)
            )
    print(f" {atom_type_} ang restraints: {len(rst)}")
    return rst
def add_dist_rst(pos,seq_,npys,atom_types,min_dises,max_dises,num_bins,constraints):
    confidence_prop = 0.5
    sep = 2
    #constraints = rosetta.core.scoring.constraints.ConstraintSet()
    for npy,atomtype,min_dis,max_dis,num_bin in zip(npys,atom_types,min_dises,max_dises,num_bins):
        anrst = gen_dist_rst(npy,seq_,pos,confidence_prop,sep,min_dis,max_dis,num_bin,atomtype)
        for    constraint in anrst:
            constraints.add_constraint(constraint)
    return constraints

def add_dih_rst(pos,seq_,npys,atom_types,constraints):
    confidence_prop = 0.5
    sep = 2
    #constraints = rosetta.core.scoring.constraints.ConstraintSet()
    for npy,atomtype in zip(npys,atom_types):
        anrst = gen_dih_rst(npy,seq_,pos,confidence_prop,sep,atomtype)
        for    constraint in anrst:
            constraints.add_constraint(constraint)
    return constraints   
def add_ang_rst(pos,seq_,npys,atom_types,constraints):
    confidence_prop = 0.5
    sep = 2
    #constraints = rosetta.core.scoring.constraints.ConstraintSet()
    for npy,atomtype in zip(npys,atom_types):
        anrst = gen_ang_rst(npy,seq_,pos,confidence_prop,sep,atomtype)
        for    constraint in anrst:
            constraints.add_constraint(constraint)
    return constraints   
def apply_cons(pos,constraints):
    csm = rosetta.protocols.constraint_movers.ConstraintSetMover()
    csm.constraint_set(constraints)
    csm.add_constraints(True)
    csm.apply(pos)




def main(npy_file,fastafile,outfile):
    init()
    npy = np.load(npy_file,allow_pickle=True).item()
    seq = (open(fastafile).readlines()[1]).strip()
    

    #
    #sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("rna/denovo/rna_lores_with_rnp_aug.wts")
    sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015")
    #sf = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("stepwise/rna/rna_res_level_energy4.wts")
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)
    
    min_mover = MinMover(mmap, sf, "lbfgs_armijo_nonmonotone", 0.0001, True)
    sf.set_weight(rosetta.core.scoring.coordinate_constraint , 1)
    sf.set_weight(rosetta.core.scoring.rna_torsion , 5)
    sf.set_weight(rosetta.core.scoring.rna_sugar_close , 2)
    sf.set_weight(rosetta.core.scoring.base_pair_constraint , 1)
    sf.set_weight(rosetta.core.scoring.atom_pair_constraint , 10)
    sf.set_weight(rosetta.core.scoring.dihedral_constraint , 6)
    sf.set_weight(rosetta.core.scoring.angle_constraint , 6)
    min_mover.max_iter(10000)
    repeat_mover = RepeatMover(min_mover, 5)

    init_pos = init_structure(seq)
    npys = [npy['pp'],npy['cc'],npy['nn']]
    min_dises=[2,2,2]
    max_dises=[30,24,18]
    num_bins=[56,44,32]
    atom_types=['P','C','N']

    constraints = rosetta.core.scoring.constraints.ConstraintSet()
    constraints=add_dist_rst(init_pos,seq,npys,atom_types,min_dises,max_dises,num_bins,constraints)
    npys = [npy['pccp'],npy['cnnc'],npy['pnnp']]
    atom_types=['PCCP','CNNC','PNNP']
    constraints=add_dih_rst(init_pos,seq,npys,atom_types,constraints)
    apply_cons(init_pos,constraints)
    npys = [npy['pcc'],npy['cnn'],npy['pnn']]
    atom_types=['PCC','CNN','PNN']
    constraints=add_ang_rst(init_pos,seq,npys,atom_types,constraints)
    apply_cons(init_pos,constraints)
    # rna_min_options = rosetta.core.import_pose.options.RNA_MinimizerOptions()
    # rna_min_options.set_max_iter(10)
    # rna_minimizer = rosetta.protocols.rna.denovo.movers.RNA_Minimizer(rna_min_options)
    # rna_minimizer.apply(init_pos)

    ####phase two#####

    repeat_mover.apply(init_pos)


    sf.show(init_pos)
    init_pos.dump_pdb(outfile)

if __name__ == "__main__":
    fastafile,outfile = sys.argv[1],sys.argv[3]
    npy_file = sys.argv[2]
    main(npy_file,fastafile,outfile)