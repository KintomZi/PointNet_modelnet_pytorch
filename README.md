PointNet_modelnet  
基于原作者yanx27的代码调试复现，原作者整合整个point的相关代码，([原作者网址](https://github.com/yanx27/Pointnet_Pointnet2_pytorch?tab=readme-ov-file))  

我单独将pointnet的代码筛选出来，并且成功复现(仅包括分类部分，不包括分割)  
作者的modelnet的数据进行了一次预处理，所以从([modelnet官网](https://modelnet.cs.princeton.edu/))下载的数据是不能运行的.  
应该是([原作者提供的数据](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip))直接下载modelnet40_normal_resampled  

关于PointNet的文献来源于 ([文献网址](https://arxiv.org/pdf/1612.00593))  


下图是作者的思路流程图  
![classification Network](https://github.com/user-attachments/assets/fb97a61d-f148-4b24-a81f-5cda81c5887a)

通过对代码的逐步研读，对作者的思路流程加以详细说明
![classificationNetworkk](https://github.com/user-attachments/assets/4793dc97-e5d4-420d-8acd-6cc658849404){width: 80%}

![T-Net](https://github.com/user-attachments/assets/f0d5ef61-6a45-4dbf-9235-fcc1e837d03c)
