# Muti-Consistensy
This is an official implement for Learning Multiscale Consistency for Self-supervised Electron Microscopy Instance Segmentation (ICASSP 24)

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
