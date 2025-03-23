import os
# 设置环境变量，允许重复加载KMP库，避免某些库加载冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
# 从model包的cnn模块中导入SimpleCNN类
from model.cnn import SimpleCNN

# 生成一个随机的输入张量，形状为(32, 3, 224, 224)
# 32表示批量大小，3表示通道数，224x224表示图像的高度和宽度
x = torch.randn(32, 3, 224, 224)
# 实例化SimpleCNN模型，设置分类数为4
model = SimpleCNN(num_class=4)
# 将输入张量传入模型进行前向传播，得到输出
output = model(x)
# 打印输出张量的形状
print(output.shape)