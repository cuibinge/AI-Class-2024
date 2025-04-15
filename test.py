import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==========================
# 配置参数（可通过函数传入或硬编码）
# ==========================
DATA_DIR = r"D:\python__pycharm\cnn-TCR\GTSRB-Training_fixed\GTSRB\Training"
BATCH_SIZE = 32
VAL_RATIO = 0.2
DEVICE = 'cpu'
RANDOM_SEED = 42

# ==========================
# 数据增强（通用配置）
# ==========================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3337, 0.3064, 0.3171],
        std=[0.2672, 0.2564, 0.2629]
    )
])

# ==========================
# CNN模型定义
# ==========================
class GTSRBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 43)
        )

    def forward(self, x):
        return self.layers(x)

# ==========================
# 独立验证函数（核心）
# ==========================
def validate_model(
        model,          # 已训练的模型
        data_loader,    # 验证集数据加载器
        criterion,      # 损失函数
        device=DEVICE  # 计算设备
):

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(data_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100*correct/total:.2f}%"
            })

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f"验证完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
    return avg_loss, accuracy

# ==========================
# 训练函数（调用独立验证函数）
# ==========================
def train():
    torch.manual_seed(RANDOM_SEED)
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int((1 - VAL_RATIO) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, BATCH_SIZE*2, shuffle=False, num_workers=0)

    model = GTSRBModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # 示例训练轮次
        # 训练阶段
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 调用独立验证函数
        print(f"\nEpoch {epoch+1} 训练损失: {running_loss/len(train_loader):.4f}")
        validate_model(model, val_loader, criterion)  # 关键调用

# ==========================
# 独立测试示例（非训练场景）
# ==========================
def standalone_test():
    """
    独立测试场景：加载模型后直接验证
    """
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到")

    # 准备测试数据加载器（需提前构建）
    test_dataset = datasets.ImageFolder(
        r"D:\python__pycharm\cnn-TCR\GTSRB-Fixed\GTSRB\Testing\Images",
        transform=transform
    )
    test_loader = DataLoader(test_dataset, BATCH_SIZE*2, num_workers=0)

    model = GTSRBModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()

    # 调用独立验证函数（测试集评估）
    validate_model(model, test_loader, criterion)

# ==========================
# 主程序入口
# ==========================
if __name__ == "__main__":
    try:
        train()  # 训练时调用验证
        # standalone_test()  # 独立测试时调用（取消注释）
    except KeyboardInterrupt:
        print("\n操作中断")
    except Exception as e:
        print(f"错误: {str(e)}")