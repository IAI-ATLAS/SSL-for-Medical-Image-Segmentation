export RESULT_DIR="/home/ws/sc1357/data/dma_spheroid_new"
export CFG_BASE="/home/ws/sc1357/projects/devel/src/dma-spheroid-bf/DLIP/experiments/configurations/dma_sph_bf/cfg_dma_sph_bf_general.yaml"
export CFG_FILE="/home/ws/sc1357/projects/devel/src/dma-spheroid-bf/DLIP/experiments/configurations/dma_sph_bf/cfg_dma_sph_bf_train.yaml"

python train.py --config_files "\
$CFG_BASE \
$CFG_FILE\
" \
--result_dir $RESULT_DIR