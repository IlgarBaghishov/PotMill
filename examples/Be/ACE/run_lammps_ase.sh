#!/bin/bash
source /users/baghishov/.bashrc

############### This is important for other programs ##########################
echo $FLUX_PMI_LIBRARY_PATH
# The FLUX_PMI_LIBRARY_PATH variable is always created under a flux instance (flux start).
PMIPATH=$(dirname $FLUX_PMI_LIBRARY_PATH)
# This is to stack LD_LIBRARY_PATH exports to look at the conda environment and flux pmi paths
# Suggested by Danny /https://flux-framework.readthedocs.io/en/latest/tutorials/lab/coral2.html 
# BOTH exports are needed for pretty much any MPI process under flux
CONDA_LD="${CONDA_PREFIX}/lib/"
export TMP_LD_LIBRARY_PATH=$CONDA_LD:$PMIPATH:$LD_LIBRARY_PATH


flux run -n 1 -c 1 -g 0 --env=LD_LIBRARY_PATH=${TMP_LD_LIBRARY_PATH} /users/baghishov/codes/lammps/build-fitsnap/lmp