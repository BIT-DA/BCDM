export NGPUS=4
# train on source data - one classifier
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR /data1/TL/DA-Seg/FADA/results/src_r101_try/ END 0

# train on source data - rwo classifier
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src_bcdm.py -cfg configs/deeplabv2_r101_src.yaml OUTPUT_DIR /data1/TL/DA-Seg/FADA/results/src_r101_two/ END 0


# train with fine-grained adversarial alignment
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_bcdm.py -cfg configs/deeplabv2_r101_bcdm.yaml OUTPUT_DIR /data1/TL/DA-Seg/FADA/results/bcdm resume /data1/TL/DA-Seg/FADA/results/src_r101_try/model_iter020000.pth END 0