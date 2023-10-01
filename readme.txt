		INSTALLATION AND IMPLEMENTATION OF DRfold
(Copyright 2022 by Zhang Lab and Freddolino lab, University of Michigan, All rights reserved)
DRfold can be used under The MIT License 
!! This version does not include trained models, users need to manually download them (see How to install the DRfold? ).

1. What is DRfold?
    DRfold (Deep RNA fold) is a program for RNA tertiary structure prediction based on deep learning potentials. 

2. How to install the DRfold?
    1) Third-party softwares
		1. PETfold (https://rth.dk/resources/petfold/download.php) # Please install PETfold v2.0, we thank juychen and luwei0917 for the test and feedback!
		2. The ViennaRNA Package (https://www.tbi.univie.ac.at/RNA/)
		3. Arena (https://github.com/pylelab/Arena)# If bin/Arena does not work, try to compile from thsi source

    2) Prerequisites
	1. Linux systems
        2. Python3 with numpy and scipy 
        3. pytorch (https://pytorch.org/get-started/locally/)
           Anaconda is recommended to set up python environments
	4. OpenMM (https://openmm.org/)

    3) Localization (We thank Martinovic Ivona for improving the document)
	1. Modify DRfold/scripts/Features.py to set 'petfoldbin' and 'ViennaRNAbin' properly, according to
			your own installation.(set the variables to executable files, not just bin/ folder, 
									meaning '/home/username/ViennaRNA/bin/RNAfold' and 
									'/home/username/PETfold/bin/PETfold').
	2. Modify the PETFOLDBIN path in DRfold/DRfold.sh (Line 6)
	3. Modify the PYTHON path in DRfold/DRfold.sh (Line 7)
	4. Move the compile Arena bin file to bin/, (if the built-in version does not work)
	4. Set executable permission to the bin file in DRfold/bin/.
		
    4) Model file downloads
	1. Download https://zhanggroup.org/DRfold/DRfold_models.zip and unzip the file
	2. Simply merge the files in unzipped 'DRfold' to the current path. Now downloaded model files shouold be available under DRfold/DeepE2EPotential/model and 
			DRfold/DeepGeoPotential/config*/model

3. Bug report:
    Please visit https://zhanglab.ccmb.med.umich.edu/bbs/?q=forum/2

4. How to run the DRfold?
	bash DRfold.sh [fastafile] [outputdir]
		fastafile: The input fastafile containing the query sequence
		outputdir:  Output file dir
	The running time for a regular RNA seqeunce (~50 nts) should be around 1 hour
	

5. Example
	You can test by fowllowing commands:
		' bash DRfold.sh test/seq.fasta test/DRfold_out '
	It will generate several files in less than 2 hours:
		' DPR.pdb ': is the final results of DRfold
		Other by-products:
		' DPR_{0~5}.pdb '       : raw structures from 6 end-to-end models (numbers are not related to the rankings) 
		' geo.npy '				: averaged geometry predictions from 3 geometry models, can be accessed by "np.load('geo.npy',allow_pickle=True).item()"

Reference
Yang Li, Chengxin Zhang, Chenjie Feng, Robin Pearce, Peter L. Freddolino, Yang Zhang.
Integrating end-to-end learning with deep geometrical potentials for ab initio RNA structure prediction, submitted.
