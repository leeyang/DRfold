#!/bin/bash
IN="$1"                # input.fasta
WDIR=`realpath -s $2`  # working folder


full_path=$(realpath $0)
 
dir_path=$(dirname $full_path)

if [ ! -s $WDIR/seq_ss.npy ]
then
    echo "Generating ss features"
    python $dir_path/Feature.py $(realpath $IN) $WDIR/seq_ss
fi

if [ ! -s $WDIR/exp_13.npy ]
then
    echo "Predicting exp_13"
    python $dir_path/exp_13/predict.py $(realpath $IN)  $WDIR/seq_ss.npy $WDIR/exp_13
fi

if [ ! -s $WDIR/exp_14.npy ]
then
    echo "Predicting exp_14"
    python $dir_path/exp_14/predict.py $(realpath $IN)  $WDIR/seq_ss.npy $WDIR/exp_14
fi

if [ ! -s $WDIR/exp_16.npy ]
then
    echo "Predicting exp_16"
    python $dir_path/exp_16/predict.py $(realpath $IN)  $WDIR/seq_ss.npy $WDIR/exp_16
fi

if [ ! -s $WDIR/exp.npy ]
then
    echo "Combining"
    python $dir_path/combine.py $WDIR/exp $WDIR/exp_14.npy $WDIR/exp_13.npy $WDIR/exp_16.npy
fi


# if [ ! -s $WDIR/fold.pdb ]
# then
#     echo "Folding"
#     /home/liyangum/amino/miniconda3/bin/python $dir_path/fold.py $(realpath $IN) $WDIR/exp.npy  $WDIR/fold.pdb
# fi