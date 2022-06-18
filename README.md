# DiSparse: Disentangled Sparsification for Multitask Model Compression

This repository is for DiSparse method introduced in the following paper accepted by CVPR2022:

DiSparse: Disentangled Sparsification for Multitask Model Compression\
Xinglong Sun, Ali Hassani, Zhangyang Wang, Gao Huang, Humphrey Shi

## Introduction
Despite the popularity of Model Compression and Multitask Learning, how to effectively compress a multitask model has been less thoroughly analyzed due to the challenging entanglement of tasks in the parameter space. In this paper, we propose DiSparse, a simple, effective, and first-of-its-kind multitask pruning and sparse training scheme. We consider each task independently by disentangling the importance measurement and take the unanimous decisions among all tasks when performing parameter pruning and selection. Our experimental results demonstrate superior performance on various configurations and settings compared to popular sparse training and pruning methods. Besides the effectiveness in compression, DiSparse also provides a powerful tool to the multitask learning community. Surprisingly, we even observed better performance than some dedicated multitask learning methods in several cases despite the high model sparsity enforced by DiSparse. We analyzed the pruning masks generated with DiSparse and observed strikingly similar sparse network architecture identified by each task even before the training starts. We also observe the existence of a "watershed" layer where the task relatedness sharply drops, implying no benefits in continued parameters sharing. 

<div align="center">
  <img src="Figs/flowchart_resize.png" width="100%">
  Overview of our method.
</div>

## Prerequisites
### Datasets
For NYU-V2, please download here:
https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view
For CityScapes, please download here:
https://drive.google.com/file/d/1WrVMA_UZpoj7voajf60yIVaS_Ggl0jrH/view
For Tiny-Taskonomy, please refer to the following page for instructions:
https://github.com/StanfordVL/taskonomy/tree/master/data

## Train

## Test

## Results
<div align="center">
  <img src="Figs/NYU_res.png" width="100%">
  Results on NYU-v2.
</div>
<div align="center">
  <img src="Figs/cityscape_res.png" width="100%">
  Results on Cityscapes.
</div>
<div align="center">
  <img src="Figs/taskonomy_res.png" width="100%">
  Results on Tiny-Taskonomy.
</div>

## Citations
