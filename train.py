import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
# filepath: c:\Users\ADsjfk\Desktop\PyTorch\train.py
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 设置设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据的均值和标准差
])

# 加载数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=6280, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

# 2. 定义神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 输入通道1，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层（10个类别）

    def forward(self, x):
        x = torch.relu(self.conv1(x))     # [B, 32, 26, 26]
        x = torch.relu(self.conv2(x))      # [B, 64, 24, 24]
        x = torch.max_pool2d(x, 2)         # [B, 64, 12, 12]
        x = self.dropout(x)
        x = x.view(-1, 64*12*12)           # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = CNN().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 5. 测试函数
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

    # 显示部分图片及预测结果（这里取测试集中的一个batch）
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True).squeeze()
    
    # 将数据转移回 CPU
    data = data.cpu()
    preds = preds.cpu()
    target = target.cpu()

    # 绘制前10张图片及它们的预测和真实标签
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for idx in range(10):
        axes[idx].imshow(data[idx].squeeze(), cmap='gray')
        axes[idx].set_title(f'预测:{preds[idx].item()}\n真实:{target[idx].item()}')
        axes[idx].axis('off')
    plt.show(block=False)
    plt.pause(3)  # 暂停，可根据需要调整时间
    plt.close()
# 6. 开始训练
for epoch in range(1, 21):  # 训练10个epoch
    train(epoch)
    test()

# 7. 保存模型
torch.save(model.state_dict(), "mnist_cnn.pth")
print("模型已保存")