# Bi-Classifier Diversity Maximization for Unsupervised Domain Adaptation
This is a [pytorch](http://pytorch.org/) implementation of [BCDM](https://arxiv.org/pdf/2008.01677).
### Prerequisites
- Python 3.6
- Pytorch 1.2.0
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0
### Step-by-step installation

```bash
conda create --name fada -y python=3.6
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
python test.py -cfg configs/deeplabv2_r101_bcdm.yaml resume g2c_r101_model_iter020000.pth
```
Our pretrained model is available at [here](https://github.com/BIT-DA/BCDM/releases).

#### Tip: For those who are interested in how performance change during the process of adversarial training, test_bcdm.py also accepts directory as the input and the results will be stored in a csv file.

### Visualization results



### Acknowledge
Some codes are adapted from [FADA](https://github.com/JDAI-CV/FADA). We thank them for their excellent projects.

### Citation
If you find this code useful please consider citing
```
@InProceedings{Li_2020,
  author = {Li, Shuang and Lv, Fangrui and Xie, Binhui, Liu, Harold Chi},
  title = {Bi-Classifier Diversity Maximization for Unsupervised Domain Adaptation},
} 
```


### Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn

or describe your problem in Issues.