#!/bin/bash
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=36
#SBATCH --mem=90000
#SBATCH --time=01:00:00
#SBATCH --partition=multiple
#SBATCH --output=job_900.out
#SBATCH --error=job_900.err
#SBATCH --export=ALL
# All this you may find in https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues
# We asked for three nodes with 40 cores and each 90GB of memory. 
# Thus partition must be multiple. Notice partitions "dev_" are 
# for development only and thus allow 4 nodes maximum and a walltime
# of 30 minutes max. output and error are where the script or the system
# writes messages. The export of all variables allows usage of those,
# see, e.g., the contents of SLURM_NTASKS
#
# These are the modules to be loaded. Do it in that order
 module load devel/python/3.10.5_gnu_12.1
 module load compiler/gnu/12.1
 module load mpi/openmpi/4.1
#
# You might have have to add the mpi4py library in case you get an error thrown.
# Just login to bwUniCluster and enter the command
# > salloc -p single -n 1 -t 120 --mem=5000 
# > module load compiler/gnu/10.2
# > module load mpi/openmpi/4.1
# > module load devel/python/3.8.6_gnu_10.2
# > python3 -m pip install --force-reinstall --no-binary :all: mpi4py
# This will fetch the sources and compile mpi4py with exactly the libraries
# you have loaded previously
#
UX_FILE="ux_500_900.npy"
UY_FILE="uy_500_900.npy"

cd ${SLURM_SUBMIT_DIR}
echo "We have a maximum of ${SLURM_NTASKS} tasks at our disposition"
mpirun --mca mpi_warn_on_fork 0 -n 900 python3 Sliding_Lid_Parallel_Cluster.py "$UX_FILE" "$UY_FILE"
# I needed to switch off warning. Give it a try if it works for you without.

