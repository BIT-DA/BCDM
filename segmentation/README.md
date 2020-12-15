## Bi-Classifier Diversity Maximization for Unsupervised Domain Adaptation

This is a [pytorch](http://pytorch.org/) implementation of BCDM.
### Prerequisites
- Python 3.6
- Pytorch >= 1.2.0
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0
### Step-by-step installation

```bash
conda create --name bcdm -y python=3.6
conda activate bcdm

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Symlink the required dataset
```bash
ln -s /path_to_gta5_dataset datasets/gta5
ln -s /path_to_cityscapes_dataset datasets/cityscapes
```

- Generate the label statics file for GTA5 Dataset by running 
```
python datasets/generate_gta5_label_info.py -d datasets/gta5 -o datasets/gta5/
```

The data folder should be structured as follows:
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── gta5/
|   |   ├── images/
|   |   ├── labels/
|   |   ├── gtav_label_info.p
│   └── 			
...
```



### Train
We provide the training script using 4 GPUs.
```
bash train_bcdm.sh
```

### Evaluate

```
python test_bcdm.py -cfg configs/deeplabv2_r101_bcdm.yaml resume results/bcdm/
```

**Tips: For those who are interested in how performance change during the process of training, ``test_bcdm.py`` also accepts directory as the input and the results are stored in a csv file.**



### Our trained Model

We also provide our trained models for direct evalution.

- [GTA5 -> Cityscapes](https://github.com/BIT-DA/BCDM/releases/)



