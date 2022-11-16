#!/bin/bash
#SBATCH --partition=haicore-gpu4
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=152
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marcel.schilling@kit.edu
#SBATCH --error=%j_error.txt
#SBATCH --output=%j_output.txt
#SBATCH --job-name=cll_process
#SBATCH --constraint=LSDF

# remove all modules
module purge
module load compiler/intel/19.1 mpi/openmpi/4.0

# activate cuda
module load devel/cuda/11.2

export CKPT_PATH="/lsdf/kit/iai/projects/iai-aida/Daten_Schilling/trained_ann/2022_02_17_DMA_FL_CLL/dnn_weights.ckpt"
export CFG_FILE="/home/hk-project-sppo/sc1357/devel/dma-cll-nuclei/DLIP/experiments/configurations/dma_cll/cfg_dma_cll_inf.yaml"

source /home/hk-project-sppo/sc1357/miniconda3/etc/profile.d/conda.sh
conda activate dma_cll

cd /home/hk-project-sppo/sc1357/devel/dma-cll-nuclei/DLIP/deployment


export RAW_DATA_PATH="/lsdf/kit/ibcs/projects/levkin-screening/KIT_Future_Fields/Markus_Reischl/Image_Analysis_CLL_Cells/2021.06.15_renamed_SU-DHL4_Sample_6"
CUDA_VISIBLE_DEVICES=0 python pipeline.py --raw_data_path $RAW_DATA_PATH --ckpt_path $CKPT_PATH --cfg_file_path $CFG_FILE &

export RAW_DATA_PATH="/lsdf/kit/ibcs/projects/levkin-screening/KIT_Future_Fields/Markus_Reischl/Image_Analysis_CLL_Cells/2021.06.22_Sample_9_Patient09"
CUDA_VISIBLE_DEVICES=1 python pipeline.py --raw_data_path $RAW_DATA_PATH --ckpt_path $CKPT_PATH --cfg_file_path $CFG_FILE &

export RAW_DATA_PATH="/lsdf/kit/ibcs/projects/levkin-screening/KIT_Future_Fields/Markus_Reischl/Image_Analysis_CLL_Cells/2021.06.24_Sample_10_Patient10"
CUDA_VISIBLE_DEVICES=2 python pipeline.py --raw_data_path $RAW_DATA_PATH --ckpt_path $CKPT_PATH --cfg_file_path $CFG_FILE 


wait < <(jobs -p)