#!/bin/bash
#SBATCH --partition=haicore-gpu4
#SBATCH --time=48:00:00
#SBATCH --ntasks=152
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcel.schilling@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=semi_super
#SBATCH --constraint=LSDF

export CFG_FILE="/home/hk-project-sppo/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/ssl/isic_dermo/derma_ssl.yaml"
export RESULT_DIR="/home/hk-project-sppo/sc1357/data/ssl"
export SWEEPID="kit-iai-ibcs-dl/derma_ssl/2i6vo99k"

# remove all modules
module purge
module load compiler/intel/19.1 mpi/openmpi/4.0

# activate cuda
module load devel/cuda/11.2

# activate conda env
source /home/hk-project-sppo/sc1357/miniconda3/etc/profile.d/conda.sh
conda activate env_ssl

# move to script dir
cd /home/hk-project-sppo/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/scripts

# start train
mpirun \
    --display-map \
    --display-allocation \
    --map-by ppr:2:socket:pe=19 \
    bash -c '
    export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK};
    unset $(printenv | grep -e ^OMPI -e ^SLURM -e ^PMIX | cut -f 1 -d=);
    unset KMP_AFFINITY
    export OMP_NUM_THREADS=19
    export MKL_NUM_THREADS=19
    wandb agent $SWEEPID'
