import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.utils.data  # 导入PyTorch的数据加载模块
import torch.nn.functional as F  # 导入PyTorch的函数形式的神经网络模块
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer  # 导入PointNet相关的工具函数


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        """
        PointNet模型定义

        Args:
            k (int): 分类数，默认为40
            normal_channel (bool): 是否使用法线通道，默认为True
        """
        super(get_model, self).__init__()  # 调用父类构造函数
        if normal_channel:
            channel = 6  # 如果使用法线信息，通道数为6
        else:
            channel = 3  # 否则通道数为3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)  # PointNet编码器
        self.fc1 = nn.Linear(1024, 512)  # 第一个全连接层，输入维度1024，输出维度512
        self.fc2 = nn.Linear(512, 256)  # 第二个全连接层，输入维度512，输出维度256
        self.fc3 = nn.Linear(256, k)  # 第三个全连接层，输入维度256，输出维度k
        self.dropout = nn.Dropout(p=0.4)  # Dropout层，防止过拟合
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化层，对第一个全连接层的输出进行归一化处理
        self.bn2 = nn.BatchNorm1d(256)  # 批归一化层，对第二个全连接层的输出进行归一化处理
        self.relu = nn.ReLU()  # 激活函数，ReLU函数

    def forward(self, x):
        """
        前向传播函数

        Args:
            x (torch.Tensor): 输入数据(点云)，维度为(batch_size, num_points, channel)

        Returns:
            torch.Tensor: 分类预测结果，维度为(batch_size, k)
            torch.Tensor: 特征变换矩阵的特征，用于正则化损失
        """
        x, trans, trans_feat = self.feat(x)  # 使用PointNet编码器提取特征

        x = F.relu(self.bn1(self.fc1(x)))  # 第一个全连接层，ReLU激活函数
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))  # 第二个全连接层，带有Dropout和ReLU激活函数
        x = self.fc3(x)  # 第三个全连接层，不带激活函数
        x = F.log_softmax(x, dim=1)  # 对最终输出进行log_softmax处理（对softmax函数进行log处理），用于分类任务
        # 其作用是获取一系列判定值，选择其中的一个最值作为类判断结果
        return x, trans_feat  # 返回预测结果和特征变换矩阵的特征


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        """
        损失函数定义

        Args:
            mat_diff_loss_scale (float): 特征变换矩阵差异损失的比例，默认为0.001
        """
        super(get_loss, self).__init__()  # 调用父类构造函数
        self.mat_diff_loss_scale = mat_diff_loss_scale  # 特征变换矩阵差异损失的比例参数

    def forward(self, pred, target, trans_feat):
        """
        前向传播函数

        Args:
            pred (torch.Tensor): 模型预测结果，维度为(batch_size, k)
            target (torch.Tensor): 真实标签，维度为(batch_size,)
            trans_feat (torch.Tensor): 特征变换矩阵的特征，用于正则化损失

        Returns:
            torch.Tensor: 总损失值
        """
        loss = F.nll_loss(pred, target)  # 计算负对数似然损失
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)  # 计算特征变换矩阵的正则化损失

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale  # 总损失为分类损失加上特征变换矩阵正则化损失
        return total_loss  # 返回总损失
