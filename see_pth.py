import torch
# 查看训练后的参数文件

pthfile = r'/root/autodl-tmp/Code_zjd/Pointnet/log/train_classification/2024-07-11_22-52/checkpoints/best_model.pth'  # .pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  # 查看模型字典里面的key
    print(k)
print('value:')
for k in model:  # 查看模型字典里面的value
    print(k, model[k])
print('OVER')
