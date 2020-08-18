# Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation (ECCV 2020)
This is a [pytorch](http://pytorch.org/) implementation of [FADA](https://arxiv.org/abs/2007.09222).
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
conda activate fada

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download/808/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Symlink the required dataset
```bash
ln -s /path_to_gta5_dataset datasets/gta5
ln -s /path_to_synthia_datasetdg datasets/synthia
ln -s /path_to_cityscapes_dataset datasets/cityscapes
```

- Generate the label statics file for GTA5 and SYNTHIA Datasets by running 
```
python datasets/generate_gta5_label_info.py -d datasets/gta5 -o datasets/gta5/
python datasets/generate_synthia_label_info.py -d datasets/synthia -o datasets/synthia/
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
│   ├── synthia/
|   |   ├── RAND_CITYSCAPES/
|   |   ├── synthia_label_info.p
│   └── 			
...
```




### Train
We provide the training script using 4 t GPUs. Note that when generating pseudo labels for self distillation, the link to the pseudo label directory should be updated [here](https://github.com/JDAI-CV/FADA/blob/98336a61f0fde633c6d504972fd782688fb8bd3a/core/datasets/dataset_path_catalog.py#L25).
```
bash train_with_sd.sh
```

### Evaluate
```
python test.py -cfg configs/deeplabv2_r101_tgt_self_distill.yaml resume g2c_tgt_self_distill_r101_model_iter018000.pth
```
Our pretrained model is available via [polybox](https://polybox.ethz.ch/index.php/s/jzckTds5efxbn3n).

#### Tip: For those who are interested in how performance change during the process of adversarial training, test.py also accepts directory as the input and the results will be stored in a csv file.

### Visualization results

![Visualization](gifs/output.gif)

### Acknowledge
Some codes are adapted from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [semseg](https://github.com/hszhao/semseg). We thank them for their excellent projects.

### Citation
If you find this code useful please consider citing
```
@InProceedings{Haoran_2020_ECCV,
  author = {Wang, Haoran and Shen, Tong and Zhang, Wei and Duan, Lingyu and Mei, Tao},
  title = {Classes Matter: A Fine-grained Adversarial Approach to Cross-domain Semantic Segmentation},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020}
} 
```
