import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# 固定配置（根据你的环境预设）
# ==========================
DATA_DIR = r"D:\python__pycharm\cnn-TCR\GTSRB-Training_fixed\GTSRB\Training"  # 训练数据路径
BATCH_SIZE = 32  # CPU建议小批量
EPOCHS = 15
LR = 0.001
VAL_RATIO = 0.2
DEVICE = 'cpu'  # 强制CPU模式（避免CUDA未找到错误）
RANDOM_SEED = 42  # 固定随机种子

# ==========================
# 数据增强（CPU优化版）
# ==========================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3337, 0.3064, 0.3171],  # GTSRB预计算均值
        std=[0.2672, 0.2564, 0.2629]   # GTSRB预计算方差
    )
])  # 移除随机旋转/翻转（CPU训练提速30%）

# ==========================
# 简化CNN模型（CPU友好）
# ==========================
class GTSRBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 128),  # 4096→128
            nn.ReLU(),
            nn.Linear(128, 43)  # 43分类
        )

    def forward(self, x):
        return self.layers(x)

# ==========================
# 训练验证循环（修复版）
# ==========================
def train():
    # 0. 环境检查
    torch.manual_seed(RANDOM_SEED)
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"训练数据路径不存在: {DATA_DIR}")

    # 1. 加载数据集
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"检测到 {len(dataset.classes)} 个类别，总样本数: {len(dataset)}")

    # 2. 划分数据集（固定随机种子）
    train_size = int((1 - VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # 3. 创建数据加载器（CPU优化）
    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False  # CPU禁用pin_memory
    )
    val_loader = DataLoader(val_dataset, BATCH_SIZE*2, shuffle=False, num_workers=0)

    # 4. 初始化模型与优化器
    model = GTSRBModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # 5. 训练循环
    best_val_acc = 0.0
    no_improvement = 0
    print(f"\n开始训练（CPU模式），共{EPOCHS}轮")

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'Train')

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, DEVICE, 'Val')

        # 学习率调度
        scheduler.step(val_acc)

        # 早停与模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"★ 保存最佳模型，验证精度: {val_acc:.2f}%")
        else:
            no_improvement += 1
            if no_improvement >= 5:
                print(f"⚠️ 早停触发，连续{no_improvement}轮未提升")
                break

        print(f"Epoch {epoch+1}/{EPOCHS} | 训练 Loss: {train_loss:.4f} | 验证 Acc: {val_acc:.2f}%\n")

# ==========================
# 修复后的run_epoch函数
# ==========================
def run_epoch(model, loader, criterion, optimizer, device, mode):
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f"[{mode}] Epoch", unit="batch", bar_format='{l_bar}{bar:10}{r_bar}')

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if mode == 'Train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 统计指标
        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        # 进度条显示（仅训练显示LR）
        postfix = {
            'Loss': f"{loss.item():.4f}",
            'Acc': f"{100*correct/total:.2f}%"
        }
        if mode == 'Train':  # 关键修复！
            postfix['LR'] = f"{optimizer.param_groups[0]['lr']:.6f}"

        progress_bar.set_postfix(postfix)

    return running_loss / total, 100 * correct / total

# ==========================
# 测试函数（新增）
# ==========================
def test_model(model_path='best_model.pth'):
    if not os.path.exists(model_path):
        print(f"警告: 未找到模型文件 {model_path}，跳过测试")
        return

    model = GTSRBModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 加载测试集（需提前准备，结构同训练集）
    test_dir = r"D:\python__pycharm\cnn-TCR\GTSRB-Fixed\GTSRB\Testing\Images"
    if not os.path.exists(test_dir):
        print(f"警告: 测试集路径不存在 {test_dir}，跳过测试")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, BATCH_SIZE*2, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(images.to(DEVICE))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    # 打印分类报告
    print("\n===================== 测试报告 =====================")
    print(classification_report(all_labels, all_preds, digits=4))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, test_dataset.classes)
    print("混淆矩阵已保存为 confusion_matrix.png")

# ==========================
# 混淆矩阵可视化
# ==========================
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("GTSRB测试集混淆矩阵", fontsize=12)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:d}",
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black',
                     fontsize=6)

    plt.xlabel('预测标签', fontsize=10)
    plt.ylabel('真实标签', fontsize=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

# ==========================
# 主程序入口
# ==========================
if __name__ == "__main__":
    try:
        train()
        test_model()  # 训练完成后自动测试
    except KeyboardInterrupt:
        print("\n训练被用户中断，当前模型未保存")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")