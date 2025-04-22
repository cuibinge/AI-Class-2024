import os
import re
import torch
import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

# 配置参数
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 1
    MODEL_NAME = 'bert-base-chinese'
    LABEL_MAP = {'sadness': 0, 'happiness': 1, 'disgust': 2,
                 'anger': 3, 'like': 4, 'surprise': 5, 'fear': 6}
    LR = 2e-5
    PROCESSED_FILE = 'processed_data.csv'

# 数据预处理
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    return ' '.join(jieba.cut(text))

# 自定义数据集
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=Config.MAX_LEN,
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

# 训练器
class EmotionTrainer:
    def __init__(self, model, train_loader, val_loader, class_weights):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(model.parameters(), lr=Config.LR)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * Config.EPOCHS
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self):
        self.model.train()
        losses = []
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())
            progress_bar.set_postfix({'loss': np.mean(losses[-10:])})

        return np.mean(losses)

    def evaluate(self):
        self.model.eval()
        val_preds = []
        val_labels = []
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_losses.append(outputs.loss.item())
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        return val_preds, val_labels, np.mean(val_losses)

# 主函数
def main():
    # 数据加载与预处理
    if os.path.exists(Config.PROCESSED_FILE):
        df = pd.read_csv(Config.PROCESSED_FILE)
    else:
        df = pd.read_csv(r'C:\Users\ADsjfk\Desktop\PyTorch\emotionget\OCEMOTION.csv', sep='\t', 
                        usecols=[1, 2], names=['text', 'label'])
        df['processed_text'] = df['text'].apply(preprocess_text)
        df['label'] = df['label'].map(Config.LABEL_MAP)
        df.to_csv(Config.PROCESSED_FILE, index=False)
    
    # 计算权重
    class_labels = np.unique(df['label'])
    class_weights = compute_class_weight('balanced', classes=class_labels, y=df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(Config.DEVICE)
    
    # 数据分割
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df['text'], df['label']))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # --- 使用 WeightedRandomSampler ---
    # 1. 计算训练集中每个样本的权重
    train_labels = train_df['label'].tolist()
    class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_labels])
    samples_weight = torch.from_numpy(samples_weight).double()

    # 2. 创建 Sampler
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # 初始化模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(Config.LABEL_MAP)
    )

    # 创建DataLoader
    train_dataset = EmotionDataset(
        train_df['processed_text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    val_dataset = EmotionDataset(
        val_df['processed_text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, shuffle=False) # 使用 sampler，关闭 shuffle
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # 训练模型
    trainer = EmotionTrainer(model, train_loader, val_loader, class_weights)
    best_accuracy = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        train_loss = trainer.train_epoch()
        val_preds, val_labels, val_loss = trainer.evaluate()
        
        accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {accuracy:.4f}")
        print(classification_report(val_labels, val_preds, target_names=Config.LABEL_MAP.keys()))
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_model.bin')
            best_accuracy = accuracy
            print("New best model saved!")
        
        # 生成并保存混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=Config.LABEL_MAP.keys(),
                    yticklabels=Config.LABEL_MAP.keys(),
                    cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.savefig(f'confusion_matrix_epoch{epoch+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nTraining complete. Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()