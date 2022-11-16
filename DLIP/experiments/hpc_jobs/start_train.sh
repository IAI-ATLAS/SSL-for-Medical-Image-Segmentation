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

export CFG_FILE="/home/hk-project-sppo/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/experiments/configurations/base_cfg/semantic/cfg_resunet_sem_seg_base.yaml"
export RESULT_DIR="/home/hk-project-sppo/sc1357/data/ssl_ref"
export SWEEPID="marcelschilling/derma_ssl/3nt5qvho"

# remove all modules
module purge

# activate cuda
module load devel/cuda/11.2

# activate conda env
source /home/hk-project-sppo/sc1357/miniconda3/etc/profile.d/conda.sh
conda activate env_ssl

# move to script dir
cd /home/hk-project-sppo/sc1357/devel/self-supervised-biomedical-image-segmentation/DLIP/scripts

# start train
CUDA_VISIBLE_DEVICES=0 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=1 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=2 wandb agent $SWEEPID &
CUDA_VISIBLE_DEVICES=3 wandb agent $SWEEPID 

wait < <(jobs -p)