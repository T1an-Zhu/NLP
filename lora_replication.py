# lora_replication.py
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

# 检查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 加载数据集
dataset = load_dataset("glue", "sst2")
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# 加载 tokenizer 和模型
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Tokenize 数据
def tokenize(batch):
    return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# 定义评估函数
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    # evaluation_strategy="steps",  
    eval_steps=200,
    report_to="none",
    load_best_model_at_end=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 训练
trainer.train()

# 可视化 loss
log_history = trainer.state.log_history
train_losses = [x['loss'] for x in log_history if 'loss' in x]

plt.plot(train_losses, label="training loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# 保存模型
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")
