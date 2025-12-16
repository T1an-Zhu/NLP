# LoRA Replication on BERT (Sequence Classification)

本项目用于复现 **LoRA（Low-Rank Adaptation）在 BERT 序列分类任务中的应用**，基于 Hugging Face Transformers 与 PEFT 框架，实现参数高效微调，并在 GPU（CUDA）环境下运行。

---

## 一、项目背景

在大规模预训练语言模型（如 BERT、GPT）上进行下游任务微调时，**全参数微调**通常具有：

* 参数量大
* 显存占用高
* 训练成本高

**LoRA（Low-Rank Adaptation）** 通过在注意力层中引入低秩矩阵，仅训练极少量新增参数，即可达到接近全参数微调的效果，是当前主流的参数高效微调方法（PEFT）。

本项目目标：

* 使用 `bert-base-uncased`
* 在文本分类任务上应用 LoRA
* 验证 LoRA 的可行性与工程流程

---

## 二、运行环境

### 1. 硬件要求

* NVIDIA GPU（支持 CUDA）
* 建议显存 ≥ 8GB

### 2. 软件环境

* 操作系统：Windows 10 / 11
* Python：3.10
* CUDA：13.x

### 3. 关键依赖版本（已验证可运行）

```txt
python==3.10.x

torch==2.9.1+cu130
torchvision==0.24.1+cu130
torchaudio==2.9.1+cu130

transformers==4.57.3
datasets==4.4.1
peft==0.18.0
accelerate>=0.21.0

numpy==1.25.0
scipy==1.11.1
scikit-learn==1.3.2
pandas==2.1.1
pyarrow==12.0.1
matplotlib
```

>  注意：本项目对 **transformers / datasets / pyarrow / numpy** 版本高度敏感，不建议随意升级或降级。

---

## 三、项目结构

```text
.
├── lora_env/                # Python 虚拟环境
├── lora_replication.py      # 主程序（LoRA 复现代码）
├── load_and_evaluate.py     # 加载训练完的模型并进行评估
├── requirements.txt         # 完整环境依赖（pip freeze 生成）
├── README.md                # 项目说明文档
```

---

## 四、核心代码说明（lora_replication.py）

### 1. 模型与设备

* 使用 `bert-base-uncased`
* 自动检测并使用 `cuda` 或 `cpu`

```python
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
).to(device)
```

首次加载时出现以下提示是**正常现象**：

```text
Some weights ... were not initialized ... ['classifier.weight', 'classifier.bias']
```

原因：

* 分类头是随机初始化的
* 需要通过下游任务训练

---

### 2. LoRA 配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
```

LoRA 特点：

* 仅在注意力层的 Q / V 投影上注入低秩矩阵
* 可训练参数占比极低（<1%）

---

### 3. 数据处理

* 使用 Hugging Face `datasets`
* 通过 `map` 批量分词

```python
train_dataset = train_dataset.map(tokenize, batched=True)
```

---

### 4. 训练配置

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="epoch",
    report_to="none"
)
```

> 注意：在当前 transformers 版本中 **不使用 `evaluation_strategy` 参数**，否则会触发 `TypeError`。

---

## 五、运行方式

### 1. 激活虚拟环境

```powershell
.\lora_env\Scripts\activate
```

### 2. 运行主程序

```powershell
python lora_replication.py
```

若看到如下输出，说明程序运行正常：

```text
Using device: cuda
Map: 100%|██████████| ...
```


---

## 六、可扩展方向

* 添加 `compute_metrics`（Accuracy / F1）
* 保存并加载 LoRA Adapter
* 对比全参数微调 vs LoRA
* 更换任务（情感分析 / 新闻分类）

---

## 七、说明

本项目用于：

* 深度自然语言处理课程作业
* LoRA 原理与工程实践学习
* Hugging Face 生态实验复现

