import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.parallel  # 导入并行计算模块
import torch.utils.data  # 导入数据处理模块
from torch.autograd import Variable  # 导入变量自动求导模块
import numpy as np  # 导入NumPy库
import torch.nn.functional as F  # 导入函数式接口模块


# 3D空间变换网络
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()  # 通过 super() 函数调用父类 nn.Module 的初始化方法，确保正确地初始化 STN3d 类。
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # 第一个1维卷积层，输入通道数为channel，输出通道数为64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 第二个1维卷积层，输入通道数为64，输出通道数为128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 第三个1维卷积层，输入通道数为128，输出通道数为1024

        self.fc1 = nn.Linear(1024, 512)  # 全连接层，输入维度为1024，输出维度为512
        self.fc2 = nn.Linear(512, 256)  # 全连接层，输入维度为512，输出维度为256
        self.fc3 = nn.Linear(256, 9)  # 全连接层，输入维度为256，输出维度为9，用于产生变换矩阵的参数
        self.relu = nn.ReLU()  # ReLU激活函数

        self.bn1 = nn.BatchNorm1d(64)  # 批标准化层，输入通道数为64
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层，输入通道数为128
        self.bn3 = nn.BatchNorm1d(1024)  # 批标准化层，输入通道数为1024
        self.bn4 = nn.BatchNorm1d(512)  # 批标准化层，输入通道数为512
        self.bn5 = nn.BatchNorm1d(256)  # 批标准化层，输入通道数为256

    def forward(self, x):
        batchsize = x.size()[0]  # 获取批大小

        x = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积、批标准化、ReLU激活
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层卷积、批标准化、ReLU激活
        x = F.relu(self.bn3(self.conv3(x)))  # 第三层卷积、批标准化、ReLU激活

        x = torch.max(x, 2, keepdim=True)[0]  # 也就是求每个文件的每个坐标维度(通道)的最大值
        # 对第2维也就是 batchsize×1024×采样点 中(第三个维度)中,进行最大化池化
        x = x.view(-1, 1024)  # 调整形状为(batchsize, 1024)

        x = F.relu(self.bn4(self.fc1(x)))  # 全连接、批标准化、ReLU激活
        x = F.relu(self.bn5(self.fc2(x)))  # 全连接、批标准化、ReLU激活
        x = self.fc3(x)  # 输出变换矩阵的参数

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)  # 创建单位矩阵，并重复batchsize次
        if x.is_cuda:  # 如果输入在GPU上运行
            iden = iden.cuda()  # 将iden转移到GPU上
        x = x + iden  # 将变换矩阵参数与单位矩阵相加

        x = x.view(-1, 3, 3)  # 调整形状为(batchsize, 3, 3)
        return x  # 返回变换矩阵


# k维空间变换网络
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)  # 第一个1维卷积层，输入通道数为k，输出通道数为64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 第二个1维卷积层，输入通道数为64，输出通道数为128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 第三个1维卷积层，输入通道数为128，输出通道数为1024
        self.fc1 = nn.Linear(1024, 512)  # 全连接层，输入维度为1024，输出维度为512
        self.fc2 = nn.Linear(512, 256)  # 全连接层，输入维度为512，输出维度为256
        self.fc3 = nn.Linear(256, k * k)  # 全连接层，输入维度为256，输出维度为k*k，用于产生变换矩阵的参数
        self.relu = nn.ReLU()  # ReLU激活函数

        self.bn1 = nn.BatchNorm1d(64)  # 批标准化层，输入通道数为64
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层，输入通道数为128
        self.bn3 = nn.BatchNorm1d(1024)  # 批标准化层，输入通道数为1024
        self.bn4 = nn.BatchNorm1d(512)  # 批标准化层，输入通道数为512
        self.bn5 = nn.BatchNorm1d(256)  # 批标准化层，输入通道数为256

        self.k = k  # 设置k值

    def forward(self, x):
        batchsize = x.size()[0]  # 获取批大小
        x = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积、批标准化、ReLU激活
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层卷积、批标准化、ReLU激活
        x = F.relu(self.bn3(self.conv3(x)))  # 第三层卷积、批标准化、ReLU激活

        x = torch.max(x, 2, keepdim=True)[0]  # 沿着最后一个维度（通道维度）进行最大化池化
        x = x.view(-1, 1024)  # 调整形状为(batchsize, 1024)

        x = F.relu(self.bn4(self.fc1(x)))  # 全连接、批标准化、ReLU激活
        x = F.relu(self.bn5(self.fc2(x)))  # 全连接、批标准化、ReLU激活
        x = self.fc3(x)  # 输出变换矩阵的参数

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)  # 创建单位矩阵，并重复batchsize次

        if x.is_cuda:  # 如果输入在GPU上运行
            iden = iden.cuda()  # 将iden转移到GPU上
        x = x + iden  # 将变换矩阵参数与单位矩阵相加

        x = x.view(-1, self.k, self.k)  # 调整形状为(batchsize, k, k)
        return x  # 返回变换矩阵


# PointNet编码器
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)  # 3D空间变换网络
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)  # 第一个1维卷积层，输入通道数为channel，输出通道数为64
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 第二个1维卷积层，输入通道数为64，输出通道数为128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 第三个1维卷积层，输入通道数为128，输出通道数为1024
        self.bn1 = nn.BatchNorm1d(64)  # 批标准化层，输入通道数为64
        self.bn2 = nn.BatchNorm1d(128)  # 批标准化层，输入通道数为128
        self.bn3 = nn.BatchNorm1d(1024)  # 批标准化层，输入通道数为1024
        self.global_feat = global_feat  # 全局特征标志
        self.feature_transform = feature_transform  # 特征变换标志
        if self.feature_transform:  # 如果需要进行特征变换
            self.fstn = STNkd(k=64)  # k维空间变换网络

    def forward(self, x):
        B, D, N = x.size()  # 获取批次、维度、点数
        trans = self.stn(x)  # 3D空间变换网络
        x = x.transpose(2, 1)  # 调换第1维和第2维，(batch_size,3,1024)->(batch_size,1024,3)
        if D > 3:  # 如果维度大于3
            feature = x[:, :, 3:]  # 暂时取出x的4列及其之后的特征
            x = x[:, :, :3]  # x为x的前3列，坐标信息
        x = torch.bmm(x, trans)  # 点积

        if D > 3:  # 如果维度大于3
            x = torch.cat([x, feature], dim=2)  # 将 拆出来的4列之后的特征 与 点积之后的结果 重新连接
        x = x.transpose(2, 1)  # 调换维度
        x = F.relu(self.bn1(self.conv1(x)))  # 1维卷积、批标准化、ReLU激活

        if self.feature_transform:  # 如果需要进行特征变换
            trans_feat = self.fstn(x)  # 特征变换
            x = x.transpose(2, 1)  # 调换维度
            x = torch.bmm(x, trans_feat)  # 点积
            x = x.transpose(2, 1)  # 调换维度
        else:
            trans_feat = None  # 特征变换为空

        pointfeat = x  # 点特征为x
        x = F.relu(self.bn2(self.conv2(x)))  # 1维卷积、批标准化、ReLU激活
        x = self.bn3(self.conv3(x))  # 1维卷积、批标准化

        x = torch.max(x, 2, keepdim=True)[0]  # 最大化池化

        x = x.view(-1, 1024)  # 调整形状为(batchsize, 1024)
        if self.global_feat:  # 如果是全局特征
            return x, trans, trans_feat  # 返回x、trans和trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)  # 重复
            return torch.cat([x, pointfeat], 1), trans, trans_feat  # 连接


# 特征变换正则化器
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]  # 获取维度
    I = torch.eye(d)[None, :, :]  # 3阶单位矩阵
    if trans.is_cuda:  # 如果输入在GPU上运行
        I = I.cuda()  # 将I转移到GPU上
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))  # 损失
    # 使运行矩阵更加趋近于正交矩阵
    return loss  # 返回损失
