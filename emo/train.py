# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # 这里改为从 torch.optim 导入
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm  # 进度条工具
from tqdm.auto import tqdm  # 自动选择适合的进度条环境
# 基本配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
MODEL_NAME = 'bert-base-chinese'
LABEL_MAP = {'sadness': 0, 'happiness': 1, 'disgust': 2,
             'anger': 3, 'like': 4, 'surprise': 5, 'fear': 6}


# 数据预处理
def preprocess_text(text):
    # 去除特殊符号和多余空格
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 结巴分词
    return ' '.join(jieba.cut(text))


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_data_loader(df, tokenizer):
    texts = df['processed_text'].tolist()
    labels = df['label'].tolist()
    dataset = EmotionDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# 数据加载与预处理保存
PROCESSED_FILE = 'processed_data.csv'
if os.path.exists(PROCESSED_FILE):
    print(f"加载已处理的文件: {PROCESSED_FILE}")
    df = pd.read_csv(PROCESSED_FILE)
else:
    print("处理原始数据并保存...")
    df = pd.read_csv('OCEMOTION.csv',
                     sep='\t',
                     header=0,
                     usecols=[1, 2],
                     names=['text', 'label'],
                     quoting=3)

    # 执行预处理
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].map(LABEL_MAP)

    # 保存处理后的数据
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"预处理数据已保存至: {PROCESSED_FILE}")

# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 初始化模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_MAP))
model.to(DEVICE)

# 创建DataLoader
train_loader = create_data_loader(train_df, tokenizer)
val_loader = create_data_loader(val_df, tokenizer)

# 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# 改进的训练函数（带进度条）
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="训练中", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 更新进度条信息
        progress_bar.set_postfix({
            'loss': loss.item(),
            'lr': scheduler.get_last_lr()[0]
        })

    return total_loss / len(data_loader)


def eval_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="验证中", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'val_loss': loss.item()})

    return total_loss / len(data_loader), np.array(predictions), np.array(true_labels)


# 训练循环（带完整进度条）
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')

    # 训练阶段
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)

    # 验证阶段
    val_loss, val_preds, val_labels = eval_model(model, val_loader)
    val_accuracy = (val_preds == val_labels).mean()

    # 打印统计信息
    print(f'\n训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}')
    print(f'验证准确率: {val_accuracy:.4f}')
    print(classification_report(val_labels, val_preds, target_names=LABEL_MAP.keys()))

    # 保存最佳模型
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_model.bin')
        best_accuracy = val_accuracy
        print("新的最佳模型已保存！")

print(f'\n训练完成，最佳验证准确率: {best_accuracy:.4f}')