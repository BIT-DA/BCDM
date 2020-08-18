export NGPUS=4
# train with fine-grained adversarial alignment
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_bcdm_woE.py -cfg configs/deeplabv2_r101_bcdm.yaml OUTPUT_DIR /data1/TL/DA-Seg/FADA/results/bcdm_woE_test resume /data1/TL/DA-Seg/FADA/results/src_r101_try/model_iter020000.pth END 0
