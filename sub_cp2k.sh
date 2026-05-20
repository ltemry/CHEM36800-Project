#!/bin/bash
#SBATCH --job-name=Au_bulk_band
#SBATCH --output=Out_Au_bulk_band.out
#SBATCH --error=Err_Au_bulk_band.err
#SBATCH --account=chem26800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=caslake

. ~/spack/share/spack/setup-env.sh
spack load cp2k %gcc@14.3.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

which cp2k.psmp
pwd
ls

mpirun -np 1 cp2k.psmp -i bulk_au_band_20_GB.inp -o bulk_au_band_20_GB.out
#mpirun -np 1 cp2k.psmp -i bulk_au_band.inp -o bulk_au_band.out
#mpirun -np 1 cp2k.psmp -i Au4_IE.inp -o Au4_charge0_E.out
