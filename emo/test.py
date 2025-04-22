# -*- coding: utf-8 -*-
import torch
import pandas as pd
import jieba
import re
from transformers import BertTokenizer, BertForSequenceClassification

# 配置参数（需与训练时保持一致）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
MODEL_PATH = 'best_model.bin'
LABEL_MAP = {
    0: 'sadness',
    1: 'happiness',
    2: 'disgust',
    3: 'anger',
    4: 'like',
    5: 'surprise',
    6: 'fear'
}


# 加载模型
def load_model():
    # 初始化模型
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=len(LABEL_MAP)
    )

    # 加载训练好的权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# 初始化组件
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = load_model()


# 预处理函数（与训练时一致）
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    return ' '.join(jieba.cut(text))


# 预测函数
def predict(text):
    # 预处理
    processed_text = preprocess_text(text)

    # Tokenization
    encoding = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # 转换为张量
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # 获取结果
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).cpu().item()
    return LABEL_MAP[pred]


# 交互式预测
if __name__ == '__main__':
    print("情感分类推理演示（输入 q 退出）")
    while True:
        text = input("\n请输入文本：")
        if text.lower() == 'q':
            break
        if len(text.strip()) == 0:
            continue

        try:
            emotion = predict(text)
            print(f"预测结果: {emotion}")
        except Exception as e:
            print(f"预测出错: {str(e)}")