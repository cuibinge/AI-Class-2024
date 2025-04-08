# -*- coding: utf-8 -*-
import os
import re
import torch
import pandas as pd
import jieba
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 配置参数
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 256
    BATCH_SIZE = 32
    EPOCHS = 1
    MODEL_NAME = 'bert-base-chinese'
    LABEL_MAP = {'sadness':0, 'happiness':1, 'disgust':2,
                 'anger':3, 'like':4, 'surprise':5, 'fear':6}
    LR = 2e-5
    ACCUM_STEPS = 2
    FGM_EPSILON = 0.3
    PROCESSED_FILE = 'enhanced_data.csv'

# 数据增强处理器
class DataProcessor:
    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        return ' '.join(jieba.cut(text))

    @staticmethod
    def augment_text(text):
        words = jieba.lcut(text)
        for i in range(len(words)):
            if np.random.rand() < 0.3:
                try:
                    syns = nearby(words[i])[0]  # 需要安装synonyms库
                    if syns: words[i] = syns[0]
                except: pass
        return ''.join(words)

    @classmethod
    def process_data(cls, file_path):
        if os.path.exists(Config.PROCESSED_FILE):
            return pd.read_csv(Config.PROCESSED_FILE)
        
        df = pd.read_csv(file_path, sep='\t', usecols=[1,2], names=['text','label'])
        df['label'] = df['label'].map(Config.LABEL_MAP)
        df['processed_text'] = df['text'].apply(lambda x: cls.augment_text(cls.preprocess_text(x)))
        df.to_csv(Config.PROCESSED_FILE, index=False)
        return df

# 改进的BERT模型
class EnhancedBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768*2, len(Config.LABEL_MAP))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled = torch.cat([hidden_states[-1][:,0], hidden_states[-2][:,0]], dim=1)
        logits = self.classifier(self.dropout(pooled))
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits}

# 对抗训练模块
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# 数据集类
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
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 训练引擎
class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(model.parameters(), lr=Config.LR)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader)*Config.EPOCHS
        )
        self.fgm = FGM(model)
        self.best_acc = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for i, batch in enumerate(progress_bar):
            inputs = {k:v.to(Config.DEVICE) for k,v in batch.items()}
            
            # 正常训练
            outputs = self.model(**inputs)
            loss = outputs['loss'] / Config.ACCUM_STEPS
            loss.backward()
            
            # 对抗训练
            self.fgm.attack(Config.FGM_EPSILON)
            outputs_adv = self.model(**inputs)
            loss_adv = outputs_adv['loss'] / Config.ACCUM_STEPS
            loss_adv.backward()
            self.fgm.restore()
            
            if (i+1) % Config.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * Config.ACCUM_STEPS
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                inputs = {k:v.to(Config.DEVICE) for k,v in batch.items()}
                outputs = self.model(**inputs)
                
                val_preds.extend(torch.argmax(outputs['logits'], dim=1).cpu().numpy())
                val_labels.extend(inputs['labels'].cpu().numpy())
        
        accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
        report = classification_report(val_labels, val_preds, target_names=Config.LABEL_MAP.keys())
        return accuracy, report

    def analyze_errors(self, val_df):
        val_df['pred'] = val_preds
        errors = val_df[val_df['label'] != val_df['pred']].sample(5)
        
        print("\n错误样本分析：")
        for idx, row in errors.iterrows():
            print(f"原文：{row['text']}")
            print(f"真实标签：{list(Config.LABEL_MAP.keys())[row['label']]}")
            print(f"预测标签：{list(Config.LABEL_MAP.keys())[row['pred']]}\n")

# 主流程
if __name__ == "__main__":
    # 数据准备
    df = DataProcessor.process_data('OCEMOTION.csv')
    
    # 分层划分数据集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, val_idx = next(sss.split(df['text'], df['label']))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    # 初始化组件
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = EnhancedBert()
    
    # 创建DataLoader
    train_loader = DataLoader(
        EmotionDataset(train_df['processed_text'].tolist(), 
                      train_df['label'].tolist(), tokenizer),
        batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        EmotionDataset(val_df['processed_text'].tolist(),
                      val_df['label'].tolist(), tokenizer),
        batch_size=Config.BATCH_SIZE
    )

    # 训练流程
    trainer = Trainer(model, train_loader, val_loader)
    
    print("开始训练...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss = trainer.train_epoch()
        val_acc, report = trainer.evaluate()
        
        print(f"验证准确率：{val_acc:.4f}")
        print(report)
        
        if val_acc > trainer.best_acc:
            torch.save(model.state_dict(), f"best_model_epoch{epoch+1}.bin")
            trainer.best_acc = val_acc
    
    # 可视化分析
    val_preds = ... # 从最后一次评估获取
    cm = confusion_matrix(val_df['label'], val_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=Config.LABEL_MAP.keys(),
               yticklabels=Config.LABEL_MAP.keys())
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')

    # 错误分析
    trainer.analyze_errors(val_df)

    print(f"训练完成，最佳准确率：{trainer.best_acc:.4f}")