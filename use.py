import torch
import torch.nn as nn
from PIL import Image  # 用于加载图片
import torchvision.transforms as transforms

# 定义相同的网络结构（必须与训练时完全一致）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 64*12*12)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# 初始化模型并加载权重
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()  # 切换到评估模式

# 图像预处理（必须与训练时相同）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 确保转换为灰度图
    transforms.Resize((28, 28)),                  # 调整尺寸
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))    # 使用相同的标准化参数
])



# 加载并预处理图像
image = Image.open("test_image.png")  # 用你的图片路径替换
image.thumbnail((28, 28)) # 缩放到指定尺寸
input_tensor = transform(image).unsqueeze(0)  # 增加batch维度 -> [1, 1, 28, 28]

with torch.no_grad():  # 禁用梯度计算
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

print(f"预测结果：数字 {prediction}")
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')
plt.title(f"Prediction: {prediction}")
plt.show()