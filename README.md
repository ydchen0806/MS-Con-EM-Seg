# MS-Con-EM-Seg
This is an official implement for Learning Multiscale Consistency for Self-supervised Electron Microscopy Instance Segmentation (ICASSP 24)
[\<Paper Link\>](https://arxiv.org/abs/2308.09917)

![The pipeline of our proposed methods](framework.png)

## Environment Setup

To set up the required environment, you can choose from the following options:

- **Using pip**:
  You can install the necessary Python dependencies from the `requirements.txt` file using the following command:

  ```bash
  pip install -r requirements.txt

We highly recommend using Docker to set up the required environment. Two Docker images are available for your convenience:

- **Using Docker from ali cloud**:
  - [**registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26**](https://registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26)
  
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/cyd_dl/monai-vit:v26
  
- **Using docker from dockerhub**:
  - [**cyd_docker:v1**](https://ydchen0806/cyd_docker:v1)
  
  ```bash
  docker pull ydchen0806/cyd_docker:v1

## Usage Guide

### 1. Pretraining
```
python pretrain.py -c pretraining_all -m train
```
### 2. Finetuning
```
python finetune.py -c seg_3d -m train -w [your pretrained path]
```

## Cite
```
@article{chen2023learning,
  title={Learning multiscale consistency for self-supervised electron microscopy instance segmentation},
  author={Chen, Yinda and Huang, Wei and Liu, Xiaoyu and Chen, Qi and Xiong, Zhiwei},
  journal={ICASSP 24},
  year={2024}
}
```
# Contact 
If you need any help or are looking for cooperation feel free to contact us. cyd0806@mail.ustc.edu.cn
