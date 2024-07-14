"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader  # 导入数据加载器
import argparse  # 解析命令行参数
import numpy as np  # 数值计算库
import os  # 操作系统接口
import torch  # PyTorch深度学习库
import logging  # 日志记录
from tqdm import tqdm  # 显示进度条的库
import sys  # 系统特定参数和函数
import importlib  # 动态导入模块

# 设置文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本文件所在目录的绝对路径
ROOT_DIR = BASE_DIR  # 设置根目录为当前脚本文件所在目录
sys.path.append(os.path.join(ROOT_DIR, 'models'))  # 将模型目录添加到系统路径中

TestFile_name = '2024-07-14_14-10'


def parse_args():
    """
    定义解析命令行参数的函数
    Returns:
        argparse.Namespace: 解析得到的命令行参数对象
    """
    parser = argparse.ArgumentParser('Testing')  # 创建参数解析器对象，命令行提示为'Testing'
    parser.add_argument('--use_cpu', action='store_true', default=False, help='使用CPU模式')  # 是否使用CPU模式的命令行参数
    parser.add_argument('--gpu', type=str, default='0', help='指定GPU设备')  # 指定GPU设备的命令行参数
    parser.add_argument('--batch_size', type=int, default=24, help='训练批次大小')  # 设置训练批次大小的命令行参数
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],
                        help='在ModelNet10/40上训练')  # 在ModelNet10/40数据集上训练的命令行参数
    parser.add_argument('--num_point', type=int, default=1024, help='点的数量')  # 每个点云中点的数量的命令行参数
    parser.add_argument('--log_dir', type=str, default=TestFile_name, help='实验根目录')  # 实验结果日志存储根目录的命令行参数，必需
    parser.add_argument('--use_normals', action='store_true', default=False, help='使用法线')  # 是否使用法线信息的命令行参数
    parser.add_argument('--use_uniform_sample', action='store_true', default=True,
                        help='使用均匀采样false/FPS采样True')  # 是否使用均匀采样的命令行参数
    parser.add_argument('--num_votes', type=int, default=3, help='使用投票聚合分类分数')  # 使用投票聚合分类分数的命令行参数
    return parser.parse_args()  # 返回解析后的命令行参数对象


def test(model, loader, num_class=40, vote_num=1):
    """
    定义测试函数
    Args:
        model: PyTorch模型
        loader: 数据加载器
        num_class: 类别数量，默认为40
        vote_num: 投票次数，默认为1

    Returns:
        float, float: 返回实例准确率和类别准确率
    """
    mean_correct = []  # 存储每批次的平均正确率
    classifier = model.eval()  # 将模型设定为评估模式
    class_acc = np.zeros((num_class, 3))  # 初始化一个数组，用于存储每个类别的准确率统计信息

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()  # 将数据移到GPU上进行计算

        points = points.transpose(2, 1)  # 转置点云数据的维度顺序，适应PyTorch模型输入要求
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()  # 初始化投票池，用于统计每个类别的投票结果

        for _ in range(vote_num):
            pred, _ = classifier(points)  # 对点云进行预测
            vote_pool += pred  # 累加预测结果到投票池中
        pred = vote_pool / vote_num  # 对投票池中的预测结果取平均，得到最终预测结果
        pred_choice = pred.data.max(1)[1]  # 选择预测概率最大的类别作为最终预测结果

        for cat in np.unique(target.cpu()):  # 遍历所有独特的类别标签
            # 计算当前类别的正确预测数量
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            # 累加当前类别的正确预测比例
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            # 当前类别的样本数量加1
            class_acc[cat, 1] += 1
        # 计算整体的正确预测数量
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # 计算最终的类别准确率和实例准确率
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]  # 计算每个类别的平均准确率
    class_acc = np.mean(class_acc[:, 2])  # 计算所有类别的平均准确率，得到类别准确率
    instance_acc = np.mean(mean_correct)  # 计算所有批次的平均正确率，得到实例准确率
    return instance_acc, class_acc  # 返回实例准确率和类别准确率


def main(args):
    """
    主函数
    Args:
        args: 命令行参数对象
    """

    def log_string(str):
        logger.info(str)  # 记录日志信息
        print(str)  # 打印日志信息到控制台

    '''超参数设置'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 设置环境变量，指定可见的GPU设备

    '''创建目录'''
    experiment_dir = 'log/' + args.log_dir  # 构建实验结果日志存储目录路径

    '''日志设置'''
    logger = logging.getLogger("Model")  # 创建Logger对象
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 设置日志格式
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)  # 创建文件处理器，将日志写入文件
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别为INFO
    file_handler.setFormatter(formatter)  # 设置文件处理器的日志格式
    logger.addHandler(file_handler)  # 将文件处理器添加到Logger对象中
    log_string('参数 ...')  # 记录和打印日志信息
    log_string(args)  # 记录和打印命令行参数信息

    '''加载数据集'''
    log_string('加载数据集 ...')  # 记录和打印日志信息
    data_path = '/root/autodl-tmp/Data_zjd/data/modelnet40_normal_resampled'  # 数据集路径

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)  # 加载测试数据集
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)  # 创建测试数据加载器

    '''加载模型'''
    num_class = args.num_category  # 获取类别数量
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]  # 获取模型名称
    model = importlib.import_module(model_name)  # 动态导入模型模块

    classifier = model.get_model(num_class, normal_channel=args.use_normals)  # 创建分类器模型
    if not args.use_cpu:
        classifier = classifier.cuda()  # 将模型移动到GPU上

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')  # 加载最佳模型检查点
    classifier.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes,
                                       num_class=num_class)  # 测试模型性能
        log_string('测试实例准确率: %f, 类别准确率: %f' % (instance_acc, class_acc))  # 记录和打印测试结果


if __name__ == '__main__':
    args = parse_args()  # 解析命令行参数
    main(args)  # 执行主函数
