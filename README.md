# BCDM
code release for "Bi-Classifier Determinacy Maximization for Unsupervised Domain Adaptation" (AAAI2021)\
[Paper (arXiv)](https://arxiv.org/abs/2012.06995)

## One-sentence description

In this paper we prove that BCDM can generate discriminative representations by encouraging target predictions to be consistent and determined enough, meanwhile, preserve the diversity of predictions in an adversarial manner, which outperforms the state-of-the-arts on a wide range
of unsupervised domain adaptation scenarios.


## Tasks
We apply BCDM to unsupervised domain adaptation (UDA) on both
image classification and semantic segmentation tasks.

Training instructions for image classification and semantic segmentation are in the README.md of [classification](https://github.com/BIT-DA/BCDM/tree/master/classification) and [segmentation](https://github.com/BIT-DA/BCDM/tree/master/segmentation) respectively.

## Citation
If you use this code for your research, please consider citing:

``` 
@inproceedings{Li21BCDM,
title = {Bi-Classifier Determinacy Maximization for Unsupervised Domain Adaptation},
author = {Shuang Li and Fangrui Lv and Binhui Xie and Chi Harold Liu and Jian Liang and Chen Qin},
booktitle = {Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},    
year = {2021}
}
```
Supplementary materials could be found in the [corresponding arXiv version](https://arxiv.org/abs/2012.06995).


## Acknowledgements
Some codes are adapted from [MCD](https://github.com/mil-tokyo/MCD_DA) and [FADA](https://github.com/JDAI-CV/FADA). We thank them for their excellent projects.

## Contact

If you have any problem about our code, feel free to contact

fangruilv@bit.edu.cn

or describe your problem in Issues.