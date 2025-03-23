# 源代码来自up主DT算法工程师前钰_原01
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model.cnn import SimpleCNN

# 使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 对训练图像进行变换
train_transformer = transforms.Compose([
    transforms.Resize([224, 224]),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 对测试图像进行变换
test_transformer = transforms.Compose([
    transforms.Resize([224, 224]),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 更新数据集路径
trainset_path = r"D:\python__pycharm\cnn卷积\dataset\COVID_19_Radiography_Dataset\train"
testset_path = r"D:\python__pycharm\cnn卷积\dataset\COVID_19_Radiography_Dataset\test"

# 验证路径是否存在
if not os.path.exists(trainset_path):
    raise FileNotFoundError(f"训练数据集路径不存在: {trainset_path}")
if not os.path.exists(testset_path):
    raise FileNotFoundError(f"测试数据集路径不存在: {testset_path}")

# 加载训练和测试数据集
trainset = datasets.ImageFolder(root=trainset_path, transform=train_transformer)
testset = datasets.ImageFolder(root=testset_path, transform=test_transformer)

# 创建数据加载器
train_loader = DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(testset, batch_size=32, num_workers=0, shuffle=False)  # 测试数据加载器

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"epoch:{epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"epoch[{epoch+1}/{num_epochs}], Train_loss: {epoch_loss:.4f}")
        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model saved with best acc")

# 评估函数
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

# 保存模型函数
def save_model(model, save_path):
    # 确认目录存在
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    num_epochs = 15  # 训练轮数
    learning_rate = 0.001  # 学习率
    num_class = 4  # 分类数
    save_path = "model_pth/best.pth"  # 模型保存路径
    model = SimpleCNN(num_class).to(device)  # 初始化模型
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器
    train(model, train_loader, criterion, optimizer, num_epochs)  # 使用训练集训练
    evaluate(model, test_loader, criterion)  # 使用测试集测试