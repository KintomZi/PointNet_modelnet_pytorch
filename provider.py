import numpy as np
# 用来记录对点云进行各种扰动的方法

def normalize_data(batch_data):
    """
    归一化批量数据，使用以原点为中心的块的坐标。

    Args:
        batch_data: BxNxC 数组，B表示批量大小，N表示点的数量，C表示维度。

    Returns:BxNxC 数组，归一化后的批量数据。B表示批量大小，N表示点的数量，C表示维度
    """

    B, N, C = batch_data.shape  # 获取批量大小、点的数量和维度
    normal_data = np.zeros((B, N, C))  # 初始化一个归一化后的数据数组
    for b in range(B):  # 遍历每个批量
        pc = batch_data[b]  # 获取当前批量数据
        centroid = np.mean(pc, axis=0)  # 计算质心
        pc = pc - centroid  # 平移点云，使质心在原点
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))  # 计算最大距离
        pc = pc / m  # 归一化
        normal_data[b] = pc  # 保存归一化后的数据
    return normal_data


def shuffle_data(data, labels):
    """
    随机打乱数据和标签。

    Args:
        data:B,N,... 数组，B表示批量大小，N表示点的数量，...表示其他维度。
        labels:B,... 数组，B表示批量大小，...表示其他维度。

    Returns:打乱后的数据，标签和打乱后的索引。

    """

    idx = np.arange(len(labels))  # 创建索引数组
    np.random.shuffle(idx)  # 随机打乱索引
    return data[idx, ...], labels[idx], idx  # 根据打乱的索引返回数据和标签


def shuffle_points(batch_data):
    """
    随机打乱每个点云中点的顺序 -- 改变FPS行为。使用相同的打乱索引用于整个批次。

    Args:
        batch_data:BxNxC 数组，B表示批量大小，N表示点的数量，C表示特征数量。

    Returns:BxNxC 数组，打乱顺序后的批量数据。B表示批量大小，N表示点的数量，C表示特征数量。

    """

    idx = np.arange(batch_data.shape[1])  # 创建索引数组
    np.random.shuffle(idx)  # 随机打乱索引
    return batch_data[:, idx, :]  # 根据打乱的索引返回数据


def rotate_point_cloud(batch_data):
    """
    随机旋转点云以增强数据集。旋转是基于形状沿着上方向的。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据。

    Returns:BxNx3 数组，旋转后的点云批量数据。

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据数组
    for k in range(batch_data.shape[0]):  # 遍历每个批量
        rotation_angle = np.random.uniform() * 2 * np.pi  # 生成随机旋转角度
        cosval = np.cos(rotation_angle)  # 计算cos值
        sinval = np.sin(rotation_angle)  # 计算sin值
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])  # 构建旋转矩阵
        shape_pc = batch_data[k, ...]  # 获取当前批量数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 进行旋转
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """
    随机绕 z 轴旋转点云数据以增强数据集。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据。

    Returns:BxNx3 数组，旋转后的点云批量数据。

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据数组
    for k in range(batch_data.shape[0]):  # 遍历每个批量
        rotation_angle = np.random.uniform() * 2 * np.pi  # 生成随机旋转角度
        cosval = np.cos(rotation_angle)  # 计算cos值
        sinval = np.sin(rotation_angle)  # 计算sin值
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])  # 构建旋转矩阵
        shape_pc = batch_data[k, ...]  # 获取当前批量数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 进行旋转
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    """
    随机旋转 XYZ 和法线点云。

    Args:
        batch_xyz_normal:batch_xyz_normal: B,N,6, 前三个通道是XYZ，后三个是法线

    Returns:B,N,6, 旋转后的 XYZ 和法线点云。

    """

    for k in range(batch_xyz_normal.shape[0]):  # 遍历每个批量
        rotation_angle = np.random.uniform() * 2 * np.pi  # 生成随机旋转角度
        cosval = np.cos(rotation_angle)  # 计算cos值
        sinval = np.sin(rotation_angle)  # 计算sin值
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])  # 构建旋转矩阵
        shape_pc = batch_xyz_normal[k, :, 0:3]  # 获取当前批量数据的XYZ
        shape_normal = batch_xyz_normal[k, :, 3:6]  # 获取当前批量数据的法线
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 旋转XYZ
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)  # 旋转法线
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    随机扰动点云和法线的小角度旋转。

    Args:
        batch_data:BxNx6 数组，原始的点云和法线批量数据。
        angle_sigma:标准差，控制扰动角度的范围，默认为0.06。
        angle_clip:截断角度，限制扰动角度的最大值，默认为0.18。

    Returns:BxNx6 数组，旋转后的点云和法线批量数据。

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据数组
    for k in range(batch_data.shape[0]):  # 遍历每个数据批次
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)  # 生成随机旋转角度并进行剪裁
        Rx = np.array([[1, 0, 0],  # 绕X轴旋转矩阵
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],  # 绕Y轴旋转矩阵
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],  # 绕Z轴旋转矩阵
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))  # 组合旋转矩阵
        shape_pc = batch_data[k, :, 0:3]  # 提取点云数据
        shape_normal = batch_data[k, :, 3:6]  # 提取法线信息
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)  # 对点云数据应用旋转矩阵
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)  # 对法线信息应用旋转矩阵
    return rotated_data  # 返回旋转后的数据


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """
    按照指定角度绕上方向旋转点云数据。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据。
        rotation_angle:旋转角度，单位为弧度。

    Returns:BxNx3 数组，旋转后的点云批量数据。

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据数组
    for k in range(batch_data.shape[0]):  # 遍历每个数据批次
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)  # 计算旋转角度的余弦值
        sinval = np.sin(rotation_angle)  # 计算旋转角度的正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],  # 绕Y轴旋转矩阵
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]  # 提取点云数据
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 应用旋转矩阵到点云数据
    return rotated_data  # 返回旋转后的数据


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """
    按照指定角度绕上方向旋转点云数据，包括法线

    Args:
        batch_data:BxNx6 数组，原始的点云和法线批量数据
        rotation_angle:旋转角度，单位为弧度

    Returns:BxNx6 数组，旋转后的点云和法线批量数据

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据，形状与输入相同
    for k in range(batch_data.shape[0]):  # 遍历每个批量
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)  # 计算旋转角度的余弦值
        sinval = np.sin(rotation_angle)  # 计算旋转角度的正弦值
        rotation_matrix = np.array([[cosval, 0, sinval],  # 构造旋转矩阵
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]  # 获取当前批次的点云数据
        shape_normal = batch_data[k, :, 3:6]  # 获取当前批次的法线数据
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)  # 旋转点云数据
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)  # 旋转法线数据
    return rotated_data  # 返回旋转后的数据


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    随机扰动点云，通过小角度旋转

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据
        angle_sigma:标准差，控制扰动角度的范围
        angle_clip:最大扰动角度，进行裁剪

    Returns:BxNx3 数组，旋转后的点云批量数据

    """

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 初始化旋转后的数据，形状与输入相同
    for k in range(batch_data.shape[0]):  # 遍历每个批量
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)  # 生成并裁剪扰动角度
        Rx = np.array([[1, 0, 0],  # 绕X轴的旋转矩阵
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],  # 绕Y轴的旋转矩阵
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],  # 绕Z轴的旋转矩阵
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))  # 组合旋转矩阵
        shape_pc = batch_data[k, ...]  # 获取当前批次的点云数据
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)  # 旋转点云数据
    return rotated_data  # 返回旋转后的数据


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """
    随机抖动点云。抖动是针对每个点的。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据
        sigma:抖动的标准差
        clip:抖动的裁剪范围

    Returns:BxNx3 数组，抖动后的点云批量数据

    """

    B, N, C = batch_data.shape  # 获取批次大小，点数量和特征数量
    assert (clip > 0)  # 确保裁剪范围大于0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -clip, clip)  # 生成并裁剪抖动值
    jittered_data += batch_data  # 将抖动值加到原始数据上
    return jittered_data  # 返回抖动后的数据


def shift_point_cloud(batch_data, shift_range=0.1):
    """
    随机平移点云。平移是针对整个点云的。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据
        shift_range:平移范围

    Returns:BxNx3 数组，平移后的点云批量数据

    """

    B, N, C = batch_data.shape  # 获取批次大小，点数量和特征数量
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    # 随机生成在-scale_low到scale_high之间缩放值数组，B默认一维数组，还可以是(B，2)等多维数组
    for batch_index in range(B):  # 遍历每个批量
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    # 缩放点云数据，!!!注意是*=符号!!!，对该批次的每个批量的三维坐标均加上一组随机常数
    return batch_data  # 返回平移后的数据


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """
    随机缩放点云。缩放是针对整个点云的。

    Args:
        batch_data:BxNx3 数组，原始的点云批量数据
        scale_low: 缩放下限
        scale_high:缩放上限

    Returns:缩放后的点云批量数据

    """

    B, N, C = batch_data.shape  # 获取批次大小，点数量和特征数量
    scales = np.random.uniform(scale_low, scale_high, B)
    # 随机生成在scale_low到scale_high之间缩放值数组，B默认一维数组，还可以是(B，2)等多维数组
    for batch_index in range(B):  # 遍历每个批量
        batch_data[batch_index, :, :] *= scales[batch_index]
        # 缩放点云数据，!!!注意是*=符号!!!，对该批次的每个批量的三维坐标均乘以一个随机常数
    return batch_data  # 返回缩放后的数据


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """
    随机丢弃点云中的点

    Args:
        batch_pc:BxNx3 数组，原始的点云批量数据
        max_dropout_ratio:最大丢弃比例

    Returns:BxNx3 数组，丢弃点后的点云批量数据

    """

    for b in range(batch_pc.shape[0]):  # 遍历每个批量
        dropout_ratio = np.random.random() * max_dropout_ratio  # 生成丢弃比例=(0.0~0.1)×max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        # 生成(batch_pc.shape[1])×1的一维数组(每个元素在0~1之间)，并每个元素与dropout_ratio比较，并记录小于等于其的索引
        nextjudge = len(drop_idx)
        if nextjudge > 0:  # 如果有点需要丢弃
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # 将丢弃点的坐标变为该点云的第一个点的坐标
    return batch_pc  # 返回丢弃点后的数据
