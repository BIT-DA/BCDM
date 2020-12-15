export NGPUS=4
# train on source data - bi-classifier
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src_bcdm.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_bcdm/

# train with BCDM
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_bcdm.py -cfg configs/deeplabv2_r101_bcdm.yaml OUTPUT_DIR results/bcdm/ resume results/src_r101_bcdm/model_iter020000.pth
