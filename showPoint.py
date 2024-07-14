import open3d as o3d  # 绘图显示库
import pickle  # 导入pickle模块，用于数据序列化和反序列化
import numpy as np
import multiprocessing as mp


def visualize_point_cloud(points: np, window_name: str, color: list, left: float, top: float):
    """
    建一个窗口显示点云

    Args:
        points: 点数据
        window_name: 窗口名字
        color: 点显示的RGB
        left: 距桌面左边的距离
        top: 距桌面顶边的距离

    Returns:

    """
    pt = o3d.geometry.PointCloud()  # 创建一个点云对象
    pt.points = o3d.utility.Vector3dVector(points)  # 将numpy数组转换为点云格式
    pt.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))  # 设置点集颜色黄色
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=700, height=500, left=left, top=top)
    vis.add_geometry(pt)
    vis.run()
    vis.destroy_window()


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


if __name__ == '__main__':
    path_process = r'D:\_All_ProgramCode\PycharmProjects\data\modelnet40_normal_resampled\airplane\airplane_0001.txt'
    npoints = 512
    if path_process == "":
        # 均匀采样
        path_load_1 = r'D:\_All_ProgramCode\PycharmProjects\Pointnet_Pointnet2_pytorch-master\log\modelnet40_train_1024pts.dat'
        # 最远点采样
        path_load_2 = r'D:\_All_ProgramCode\PycharmProjects\Pointnet_Pointnet2_pytorch-master\log\modelnet40_train_1024pts_fps.dat'
        with open(path_load_1, 'rb') as f:  # 打开保存文件
            point_set1, cls1 = pickle.load(f)  # 加载点云数据和标签
        with open(path_load_2, 'rb') as f:  # 打开保存文件
            point_set2, cls2 = pickle.load(f)  # 加载点云数据和标签
        point1 = point_set1[0]
        point2 = point_set2[0]

    else:
        point_set = np.loadtxt(path_process, delimiter=',').astype(np.float32)  # 加载点云数据
        point_set1 = farthest_point_sample(point_set, npoints)  # 使用最远点采样
        point_set2 = point_set[0:npoints, :]  # 截取前n个点
        point1 = point_set1
        point2 = point_set2

    show_point_1 = point1[:, 0:3]
    show_point_2 = point2[:, 0:3]
    if path_process != "":
        show_point_3 = np.loadtxt(path_process, delimiter=',').astype(np.float32)[:, 0:3]
        yuanLong = np.loadtxt(path_process, delimiter=',').astype(np.float32).shape[0]
        process3 = mp.Process(target=visualize_point_cloud,
                              args=(show_point_3, "原数据(共" + str(yuanLong) + "个点)", [0, 1, 0], 25, 600))

    process1 = mp.Process(target=visualize_point_cloud,
                          args=(show_point_1, "均匀采样(截选前" + str(npoints) + "个点)", [0, 0, 0], 25, 50))
    process2 = mp.Process(target=visualize_point_cloud,
                          args=(show_point_2, "最远点采样(FPS：" + str(npoints) + ")", [1, 0, 0], 750, 50))


    # 启动线程
    process1.start()
    process2.start()
    if path_process != "":
        process3.start()
        process3.join()

    # 等待进程结束
    process1.join()
    process2.join()


    print('over')
