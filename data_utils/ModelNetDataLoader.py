'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os  # 导入操作系统模块
import numpy as np  # 导入数值计算模块NumPy
import warnings  # 导入警告处理模块
import pickle  # 导入pickle模块，用于数据序列化和反序列化

from tqdm import tqdm  # 导入进度条模块tqdm
from torch.utils.data import Dataset  # 从PyTorch中导入Dataset类

warnings.filterwarnings('ignore')  # 忽略所有警告


def pc_normalize(pc):
    """
    对点云数据进行归一化处理

    Args:
        pc: 点云数据，形状为[N, D]
    Returns:
        归一化后的点云数据
    """
    centroid = np.mean(pc, axis=0)  # 计算点云数据的中心
    pc = pc - centroid  # 将点云数据平移到中心
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 计算点云数据到中心的最大距离
    pc = pc / m  # 对点云数据进行归一化处理
    return pc  # 返回归一化后的点云数据


# [0, 3], [1, 3], [4, 0], [0, 0]从第一个点开始采样3个点，第三个点是谁有待争议
def farthest_point_sample(point, npoint):
    """
    最远点采样算法

    Args:
        point: 点云数据，形状为[N, D]
        npoint: 采样点的数量
    Returns:
        centroids: 采样点的索引，形状为[npoint, D]
    """
    N, D = point.shape  # 获取点云数据的形状
    xyz = point[:, :3]  # 获取点云数据的前三列，即坐标信息
    centroids = np.zeros((npoint,))  # 初始化采样点索引
    distance = np.ones((N,)) * 1e10  # 初始化距离数组，所有值为无穷大
    farthest = np.random.randint(0, N)  # 随机选择一个初始点

    for i in range(npoint):
        centroids[i] = farthest  # 将当前点加入采样点集合
        centroid = xyz[farthest, :]  # 获取当前点的坐标
        over = (xyz - centroid)**2
        dist = np.sum(over, -1)  # 计算所有点到当前点的距离
        mask = dist < distance  # 找到 当前点到各个点之间的距离 小于 上一个点到各个点之间的距离 的点
        distance[mask] = dist[mask]  # 更新距离数组
        farthest = np.argmax(distance, -1)  # 选择距离最远的点作为新的最远点

    point = point[centroids.astype(np.int32)]  # 根据索引选择采样点
    return point  # 返回采样后的点云数据


def farthest_point_sample_Pro(points, n_samples):
    """
    最远点采样算法

    Args:
       points: 原始点云数据，形状为 (N, D)，其中 N 为点数，D 为每个点的维度
       n_samples: 需要采样的点数

    Returns:
       sp: 采样后的点，形状为 (n_samples, D)
    """
    N, D = points.shape  # 获取点的数量 N 和每个点的维度 D
    sp = np.zeros((n_samples, D))  # 初始化采样点矩阵
    fi = 0  # 随机选择第几个点作为第一个采样点

    def find_nth_largest_position(arr, n):
        """
        寻找 NumPy 数组中第 n 大的值的位置
        参数:
        - arr: 输入的 NumPy 数组
        - n: 第 n 大的位置，例如 n=1 表示最大值，n=2 表示第二大的值，依此类推

        返回:
        - position: 第 n 大值在原始数组中的位置（索引）
        """
        # 使用 argsort 方法得到数组按升序排序后的索引
        sorted_indices = np.argsort(arr)
        # 计算第 n 大的值在原始数组中的位置
        position = sorted_indices[-n]
        return position

    for i in range(n_samples):
        sp[i] = points[fi]  # 将选中的点加入采样点集合
        point = points[fi]  # 当前采样点#0 1 5 3
        dist = np.linalg.norm(points - point, axis=1)  # 计算当前采样点到所有点的距离
        fi = find_nth_largest_position(dist, i + 1)

    return sp  # 返回采样点集合


class ModelNetDataLoader(Dataset):
    "定义了一个名为 ModelNetDataLoader 的类，它是 Dataset 的子类"

    def __init__(self, root, args, split='train', process_data=False):
        """
        初始化ModelNet数据加载器

        Args:
            root: 数据集的根目录
            args: 参数对象
            split: 数据集划分类型，'train'或'test'
            process_data: 是否进行采样点的离线数据的读取与保存
        """

        self.root = root  # 设置数据集根目录
        self.npoints = args.num_point  # 设置采样点数量
        self.process_data = process_data  # 是否处理数据
        self.uniform = args.use_uniform_sample  # 是否使用均匀采样
        self.use_normals = args.use_normals  # 是否使用法线信息
        self.num_category = args.num_category  # 类别数量

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')  # 设置10类类别文件路径
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')  # 设置40类类别文件路径

        self.cat = [line.rstrip() for line in open(self.catfile)]  # 读取类别名称
        self.classes = dict(zip(self.cat, range(len(self.cat))))  # 创建类别字典,所谓的建立链表

        shape_ids = {}  # 初始化形状ID字典
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in
                                  open(os.path.join(self.root, 'modelnet10_train.txt'))]  # 读取训练集ID
            shape_ids['test'] = [line.rstrip() for line in
                                 open(os.path.join(self.root, 'modelnet10_test.txt'))]  # 读取测试集ID
        else:
            shape_ids['train'] = [line.rstrip() for line in
                                  open(os.path.join(self.root, 'modelnet40_train.txt'))]  # 读取训练集ID
            shape_ids['test'] = [line.rstrip() for line in
                                 open(os.path.join(self.root, 'modelnet40_test.txt'))]  # 读取测试集ID

        assert (split == 'train' or split == 'test')  # 断言split参数为'train'或'test'
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]  # 提取形状名称
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]  # 构建数据路径列表
        print('用于 %s 的数据集的数量达 %d 个' % (split, len(self.datapath)))  # 打印数据集大小

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            # 设置保存路径（均匀采样）
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
            # 设置保存路径（非均匀采样）

        if self.process_data:  # 如果需要处理数据
            if not os.path.exists(self.save_path):  # 如果保存路径不存在
                print('未发现已处理完的数据,开始处理点云数据...')  # 打印处理数据提示
                self.list_of_points = [None] * len(self.datapath)  # 初始化点云数据列表
                self.list_of_labels = [None] * len(self.datapath)  # 初始化标签列表
                if self.uniform:
                    print("采用最远点采样")# 使用最远点采样
                else:
                    print("截取前n个点") # 截取前n个点

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):  # 迭代数据路径
                #for index in range(0, 1):
                    fn = self.datapath[index]  # 获取文件路径
                    cls = self.classes[self.datapath[index][0]]  # 获取类别索引
                    cls = np.array([cls]).astype(np.int32)  # 转换类别索引为整数数组
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)  # 加载点云数据

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)  # 使用最远点采样
                    else:
                        point_set = point_set[0:self.npoints, :]  # 截取前n个点

                    self.list_of_points[index] = point_set  # 存储点云数据
                    self.list_of_labels[index] = cls  # 存储标签

                print("处理完毕，开始保存处理的点云数据：%s" % self.save_path)
                with open(self.save_path, 'wb') as f:  # 打开保存文件
                    pickle.dump([self.list_of_points, self.list_of_labels], f)  # 保存点云数据和标签
            else:
                print('正在读取已处理完的数据...\n路径： %s' % self.save_path)  # 打印加载数据提示
                with open(self.save_path, 'rb') as f:  # 打开保存文件
                    self.list_of_points, self.list_of_labels = pickle.load(f)  # 加载点云数据和标签
                print("读取完毕")

    def __len__(self):
        """
        返回数据集的大小

        Returns:数据集大小
        """
        return len(self.datapath)

    def _get_item(self, index):
        """
        获取指定索引的数据项

        Args:
            index:数据项索引

        Returns:
            point_set: 点云数据
            label: 标签
        """
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]  # 从列表中获取点云数据和标签
        else:
            fn = self.datapath[index]  # 获取[类别，文件路径]绑定列表
            cls = self.classes[self.datapath[index][0]]  # 获取类别索引
            label = np.array([cls]).astype(np.int32)  # 转换类别索引为整数数组
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)  # 加载点云数据

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)  # 使用最远点采样
            else:
                point_set = point_set[0:self.npoints, :]  # 截取前n个点

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 对点云数据的所有行，前三列进行归一化
        if not self.use_normals:
            point_set = point_set[:, 0:3]  # 如果不使用法线信息，则只保留前三列

        return point_set, label[0]  # 返回点云数据和标签

    def __getitem__(self, index):
        """
        获取指定索引的数据项

        Args:
            index:数据项索引
        Returns:
            point_set: 点云数据
            label: 标签

        """
        return self._get_item(index)


if __name__ == '__main__':  # 程序开始位置,如果运行此文件就执行下面的操作，不执行就相当于一个方法库
    import torch  # 导入PyTorch模块
    import argparse


    def parse_args():  # 初始化参数设置
        '''PARAMETERS'''
        parser = argparse.ArgumentParser()  # 创建解析器对象
        parser.add_argument('-n', '--num_category', default=40, type=int, choices=[10, 40],
                            help='在ModelNet10/40上训练')  # 添加类别数量选项
        parser.add_argument('--num_point', type=int, default=1024, help='点的数量')  # 添加点数量选项
        parser.add_argument('--use_normals', action='store_false', help='使用法线')  # 添加使用法线选项
        parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='使用均匀采样false/FPS采样True')
        return parser.parse_args()  # 解析参数并返回


    args = parse_args()

    # 本地路径
    # root = r'../data/modelnet40_normal_resampled' # 相对路径不好用，难得考虑变化
    # root = r'D:\_All_ProgramCode\PycharmProjects\data\modelnet40_normal_resampled'

    # 云服务器路径,！！！记得要实时更新文件
    root = '/root/autodl-tmp/Data_zjd/data/modelnet40_normal_resampled'

    data = ModelNetDataLoader(root=root, args=args, split='train', process_data=True)

    # data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')  # 创建数据加载器实例
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)  # 创建PyTorch数据加载器
    for point, label in DataLoader:  # 遍历数据加载器
        print(point.shape)  # 打印点云数据形状
        print(label.shape)  # 打印标签形状
