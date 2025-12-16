import torch
import numpy as np
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from peft import PeftModel # 导入用于加载LoRA模型的关键库
import time

# 1. 配置和数据准备
LORA_MODEL_PATH = "./lora_model" # 模型保存路径
BASE_MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16 

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Loading LoRA Adapter from: {LORA_MODEL_PATH}")

# 加载数据集和 tokenizer (用于评估)
tokenizer = BertTokenizer.from_pretrained(LORA_MODEL_PATH) # 从保存的文件夹加载tokenizer
dataset = load_dataset("glue", "sst2")
eval_dataset = dataset['validation']

# Tokenize 数据 (确保与训练时参数一致)
def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)

eval_dataset = eval_dataset.map(tokenize, batched=True)

eval_dataset = eval_dataset.remove_columns(["sentence", "idx"])
eval_dataset = eval_dataset.rename_column("label", "labels")

eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 2. 模型加载（跳过训练）

# 1. 加载原始的 BERT 基础模型
base_model = BertForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, 
    num_labels=2
)
# 2. 加载 LoRA Adapter，并将其注入到基础模型中
# 注意：PeftModel.from_pretrained 自动处理配置和权重合并
model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_PATH,
    is_trainable=False # 确保模型处于评估模式，不再训练
)

model.to(device)

# 确保模型将 LoRA 权重与基础权重合并 (可选，但推荐用于推理/评估)
# model = model.merge_and_unload() 
# *注：对于 Trainer 训练，通常在评估时不需要合并，但手动评估时可以合并以获得纯净的最终模型结构。

# 3. 手动评估阶段 (使用已加载的模型)
@torch.no_grad()
def manual_evaluate(model, dataset, batch_size):
    print("\n 执行手动评估")
    model.eval() # 设置为评估模式
    
    eval_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    end_time = time.time()
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"评估耗时: {end_time - start_time:.2f} 秒")
    return {"accuracy": accuracy}

# 执行手动评估
final_eval_results = manual_evaluate(
    model, 
    eval_dataset, 
    batch_size=BATCH_SIZE
)

# 打印最终准确率
print(" 最终评估结果:")
print(f"验证集准确率: {final_eval_results.get('accuracy'):.4f}")