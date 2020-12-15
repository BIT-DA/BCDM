
## Bi-Classifier Diversity Maximization for Unsupervised Domain Adaptation
This is a [pytorch](http://pytorch.org/) implementation of BCDM.


### Prerequisites
- Python 3.6
- Pytorch >= 1.2.0
- torchvision 
- numpy
- argparse
- random
- PIL
- CUDA >= 9.0


### Step-by-step installation
```bash
conda create --name bcdm -y python=3.6
conda activate bcdm

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

## Dataset
### VisDA-2017
VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.


### Train
Train on VisDA2017 with ResNet101
```
cd pytorch
python train_visda.py --gpu 0 --resnet 101 --train_path .../data//visda2017/clf/train --val_path .../data/visda2017/clf/validation
```

