"""
Author: Benny
Date: Nov 2019
"""

import os  # 导入操作系统模块
import sys  # 导入系统模块
import torch  # 导入PyTorch模块
import numpy as np  # 导入NumPy模块

import datetime  # 导入日期时间模块
import logging  # 导入日志模块
import provider  # 导入provider模块
import importlib  # 导入动态加载模块
import shutil  # 导入高级文件操作模块
import argparse  # 导入命令行选项解析模块

from pathlib import Path  # 导入路径模块
from tqdm import tqdm  # 导入进度条模块
from data_utils.ModelNetDataLoader import ModelNetDataLoader  # 导入ModelNet数据加载器

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在的目录
ROOT_DIR = BASE_DIR  # 设置根目录
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 将模型目录添加到系统路径中


# 点云分类有两个指标：总体的分类准确率、类平均准确率
# 要么epoch最好结果算，要么就进行检验

def parse_args():
    """
    解析命令行参数
    Returns:argparse.Namespace: 包含命令行参数的命名空间对象

    """

    parser = argparse.ArgumentParser('training')  # 创建解析器对象
    parser.add_argument('--use_cpu', action='store_true', default=False, help='使用CPU模式')  # 添加使用CPU模式选项，不使用CPU
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # 添加GPU设备选项，使用第0张GPU
    parser.add_argument('--batch_size', type=int, default=24, help='训练时的批处理大小')  # 添加批处理大小选项，一次性进入神经网络数量
    parser.add_argument('--model', default='pointnet_cls', help='模型名称 [默认: pointnet_cls]')  # 添加模型名称选项，选择使用的模型框架
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],
                        help='在ModelNet10/40上训练')  # 添加类别数量选项
    parser.add_argument('--epoch', default=2, type=int, help='训练的迭代次数')  # 添加迭代次数选项
    parser.add_argument('--learning_rate', default=0.001, type=float, help='训练中的学习率')  # 添加学习率选项
    parser.add_argument('--num_point', type=int, default=1024, help='点的数量')  # 添加点数量选项
    parser.add_argument('--optimizer', type=str, default='Adam', help='训练的优化器')  # 添加优化器选项
    parser.add_argument('--log_dir', type=str, default=None, help='实验根目录')  # 添加日志目录选项
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='衰减率')  # 添加衰减率选项
    parser.add_argument('--use_normals', action='store_true', default=False, help='使用法线')  # 添加使用法线选项
    parser.add_argument('--process_data', action='store_true', default=False, help='离线保存数据')  # 添加离线保存数据选项
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='使用均匀采样false/FPS采样True')  # 添加均匀采样选项
    return parser.parse_args()  # 解析参数并返回


def inplace_relu(m):
    """
    将ReLU激活函数设置为inplace模式

    Args:
        m:模块

    Returns:
        None
    """
    classname = m.__class__.__name__  # 获取模块的类名
    if classname.find('ReLU') != -1:  # 如果类名中包含'ReLU'
        m.inplace = True  # 将inplace属性设置为True


def test(model, loader, num_class=40):
    """
    测试模型

    Args:
        model:模型
        loader: 数据加载器
        num_class:类别数量

    Returns:
         float: 实例准确率
         float: 类别准确率
    """

    mean_correct = []  # 初始化正确率列表
    class_acc = np.zeros((num_class, 3))  # 初始化类别准确率数组
    classifier = model.eval()  # 设置模型为评估模式

    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):  # 遍历测试数据
    for j, (points, target) in enumerate(loader):
        if not args.use_cpu:  # 如果不使用CPU
            points, target = points.cuda(), target.cuda()  # 将数据移动到GPU

        points = points.transpose(2, 1)  # 转置点数据
        pred, _ = classifier(points)  # 预测
        pred_choice = pred.data.max(1)[1]  # 获取预测结果

        for cat in np.unique(target.cpu()):  # 遍历每个类别
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()  # 计算类别准确率
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])  # 累加准确率
            class_acc[cat, 1] += 1  # 累加类别数

        correct = pred_choice.eq(target.long().data).cpu().sum()  # 计算总的正确数
        mean_correct.append(correct.item() / float(points.size()[0]))  # 计算平均正确率

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]  # 计算每个类别的平均准确率
    class_acc = np.mean(class_acc[:, 2])  # 计算所有类别的平均准确率
    instance_acc = np.mean(mean_correct)  # 计算实例的平均准确率

    return instance_acc, class_acc  # 返回实例准确率和类别准确率


def main(args):
    """
    主函数，用于训练和测试模型

    Args:
        args:命令行参数对象

    Returns:
        float: 最佳实例准确率
        int: 最佳迭代次数
    """

    def log_string(str):
        """
        记录日志和打印输出

        Args:
            str: 要记录和打印的字符串

        Returns:
            None
        """
        logger.info(str)  # 记录日志
        print(str)  # 打印输出

    '''超参数设置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 默认是第0块GPU,但可能GPU出现问题或者运存已满，写在这支持报错与修改

    '''创建日志目录'''
    exp_dir = Path('./log/')  # 设置同级目录下的日志路径所在
    exp_dir.mkdir(exist_ok=True)  # 创建日志目录，只有在目录不存在时创建目录，目录已存在时不会抛出异常。

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))  # 获取当前时间字符串
    if args.log_dir is None:  # 如果未指定日志目录
        exp_dir = exp_dir.joinpath(timestr)  # 使用时间字符串作为日志目录
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)  # 使用指定的日志目录
    exp_dir.mkdir(exist_ok=True)  # 创建日志目录

    checkpoints_dir = exp_dir.joinpath('checkpoints/')  # 设置检查点目录
    checkpoints_dir.mkdir(exist_ok=True)  # 创建检查点目录

    log_dir = exp_dir.joinpath('logs/')  # 设置日志目录
    log_dir.mkdir(exist_ok=True)  # 创建日志目录

    '''日志设置'''
    args = parse_args()  # 解析命令行参数
    logger = logging.getLogger("Model")  # 创建日志记录器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))  # 创建文件日志处理器
    file_handler.setLevel(logging.INFO)  # 设置文件日志处理器的级别为INFO，事件级别(debug，info，warning，error，critical)
    file_handler.setFormatter(formatter)  # 设置文件日志处理器的格式
    logger.addHandler(file_handler)  # 添加文件日志处理器
    log_string('\n本次运行参数如下 ...')  # 记录参数日志
    log_string(args)  # 记录参数日志

    '''数据加载'''
    log_string('加载数据集 ...')  # 记录加载数据日志
    data_path = '/root/autodl-tmp/Data_zjd/data/modelnet40_normal_resampled'  # 数据路径

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train',
                                       process_data=args.process_data)  # 创建训练数据集
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test',
                                      process_data=args.process_data)  # 创建测试数据集

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    # 创建训练数据加载器，shuffle是否打乱，numworker线程数量，drop_last是否丢去不足一batch_size的样本,
    # !!!值得注意的是线程数量要小于CPU线程!!!

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)  # 创建测试数据加载器

    '''模型加载'''
    # 为了确保每次运行时的参数是可见的
    num_class = args.num_category  # 类别数量
    model = importlib.import_module(args.model)  # 动态加载模型模块
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))  # 复制模型文件到实验目录
    shutil.copy('models/pointnet_utils.py', str(exp_dir))  # 复制模型工具文件到实验目录
    shutil.copy('./train_classification.py', str(exp_dir))  # 复制训练文件到实验目录

    classifier = model.get_model(num_class, normal_channel=args.use_normals)  # 获取模型
    criterion = model.get_loss()  # 获取损失函数
    classifier.apply(inplace_relu)  # 应用inplace_relu

    '''将模型和损失函数移动到GPU'''
    if not args.use_cpu:  # 如果不使用CPU
        classifier = classifier.cuda()  # 将模型移动到GPU
        criterion = criterion.cuda()  # 将损失函数移动到GPU

    '''加载预训练模型'''
    try:
        # 确保模型训练中断后，不必重新执行，随时存储相关参数
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')  # 尝试加载预训练模型
        start_epoch = checkpoint['epoch']  # 获取预训练模型的开始迭代数
        classifier.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态字典
        log_string('使用预训练模型')  # 记录使用预训练模型日志
    except:
        log_string('没有现有模型，从头开始训练...')  # 记录从头开始训练日志
        start_epoch = 0  # 设置开始迭代数为0

    '''优化器设置'''
    if args.optimizer == 'Adam':  # 如果选择Adam优化器
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,  # 初始训练学习率
            betas=(0.9, 0.999),
            eps=1e-08,  # 防止分母为零
            weight_decay=args.decay_rate
        )  # 设置Adam优化器
    else:  # 如果选择SGD优化器，SGD随机梯度下降
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)  # 设置SGD优化器

    '''学习率调度器设置'''
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)  # 设置学习率调度器

    global_epoch = 0  # 全局迭代数
    global_step = 0  # 全局步骤数
    best_instance_acc = 0.0  # 最佳实例准确率
    best_class_acc = 0.0  # 最佳类别准确率

    '''开始训练'''
    logger.info('开始训练...')  # 记录开始训练日志
    for epoch in range(start_epoch, args.epoch):  # 遍历每个迭代
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))  # 记录当前迭代日志
        mean_correct = []  # 初始化平均正确率列表
        classifier = classifier.train()  # 设置模型为训练模式
        scheduler.step()  # 更新学习率

        # 遍历每个批次，并以tqdm显示进度条
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),smoothing=0.9):
        #for batch_id, (points, target) in enumerate(trainDataLoader):
            optimizer.zero_grad()  # 优化器，清零梯度

            # [1]标量
            # [[1,2,3],[6,7,8]]矩阵 shape是（2,3）
            # 三维以上则称之为张量

            points = points.data.numpy()  # cuda默认进去为张量，所以将张量转换为NumPy数组
            # numpy数组才能进行数据增强的操作
            points = provider.random_point_dropout(points)  # 随机丢弃点
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  # 随机缩放点云
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  # 随机平移点云
            points = torch.Tensor(points)  # 转换回 Tensor
            points = points.transpose(2, 1)  # 转置点数据，因为神经网络要求中间的数据必须是维度
            # 将张量(batch_size,样本数量,维度/通道数)->(batch_size,维度/通道数,样本数量)

            if not args.use_cpu:  # 如果不使用CPU
                points, target = points.cuda(), target.cuda()  # 将数据移动到GPU

            pred, trans_feat = classifier(points)  # 向前传播，将point传给分类器模型，返回pred预测结果，
            loss = criterion(pred, target.long(), trans_feat)  # 计算损失

            pred_choice = pred.data.max(1)[1]  # 获取各个列的最大值的位置作为预测结果
            a = target.long()
            b = a.data
            c = pred_choice.eq(b)
            d = c.cpu()
            e = d.sum()
            correct = pred_choice.eq(target.long().data).cpu().sum()  # 放到CPU上计算正确数
            mean_correct.append(correct.item() / float(points.size()[0]))
            # 计算平均正确率，将correct的tensor值数值化，再除以批次数量。最后，将结果存储到mean_correct中
            # .size与.shape一个效果，shape是torch定义下的，而size是继承下的
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            global_step += 1  # 更新全局步骤数

        train_instance_acc = np.mean(mean_correct)  # 计算训练实例准确率
        log_string('训练实例准确率: %f' % train_instance_acc)  # 记录训练实例准确率日志

        '''测试模型'''
        with torch.no_grad():  # 禁用梯度计算
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)  # 测试模型

            if instance_acc >= best_instance_acc:  # 如果实例准确率提升
                best_instance_acc = instance_acc  # 更新最佳实例准确率
                best_epoch = epoch + 1  # 更新最佳迭代数

            if class_acc >= best_class_acc:  # 如果类别准确率提升
                best_class_acc = class_acc  # 更新最佳类别准确率
            log_string('测试实例准确率: %f, 类别准确率: %f' % (instance_acc, class_acc))  # 记录测试准确率日志
            log_string('最佳实例准确率: %f, 类别准确率: %f' % (best_instance_acc, best_class_acc))  # 记录最佳准确率日志

            if instance_acc >= best_instance_acc:  # 如果实例准确率提升
                logger.info('保存模型...')  # 记录保存模型日志
                savepath = str(checkpoints_dir) + '/best_model.pth'  # 设置保存路径
                log_string('保存于 %s' % savepath)  # 记录保存路径日志
                state = {
                    'epoch': best_epoch,  # 迭代数
                    'instance_acc': instance_acc,  # 实例准确率
                    'class_acc': class_acc,  # 类别准确率
                    'model_state_dict': classifier.state_dict(),  # 模型状态字典
                    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态字典
                }
                torch.save(state, savepath)  # 保存模型
            global_epoch += 1  # 更新全局迭代数

    logger.info('训练结束...')  # 记录训练结束日志


if __name__ == '__main__':
    args = parse_args()  # 解析命令行参数
    main(args)  # 执行主函数
