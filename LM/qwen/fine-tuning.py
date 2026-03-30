import torch
import json
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os

# 本地模型路径
model_name = "/home/extra1T/wpw/LM/qwen/qwen2.5-7B-Instruct"

# ----------------------
# 加载 Tokenizer
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------
# 加载模型（无量化）
# ----------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    trust_remote_code=True,
    device_map="auto"
)

# ----------------------
# 加载数据集
# ----------------------
train_dataset = load_dataset("json", data_files="/home/extra1T/wpw/LM/qwen/dataset/demo/train_90.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="/home/extra1T/wpw/LM/qwen/dataset/demo/eval_10.jsonl", split="train")

# ----------------------
# LoRA 设置
# ----------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# ----------------------
# 训练参数
# ----------------------
training_args = SFTConfig(
    output_dir="./scitldr_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    bf16=True,
    fp16=False,
    optim="paged_adamw_8bit",
    report_to="none",
    save_strategy="no",
)

# ----------------------
# 训练器
# ----------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

# ===================== 【新增】日志保存功能 =====================
# 创建日志文件路径
log_file = os.path.join(training_args.output_dir, "training_log.txt")

# 确保文件夹存在
os.makedirs(training_args.output_dir, exist_ok=True)

# 打开日志文件，准备写入
with open(log_file, "w", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write(f"训练开始时间：{time.ctime()}\n")
    f.write(f"模型路径：{model_name}\n")
    f.write(f"训练集大小：{len(train_dataset)}\n")
    f.write(f"验证集大小：{len(eval_dataset)}\n")
    f.write("="*50 + "\n\n")

# 定义日志回调函数
def log_training_logs(logs):
    with open(log_file, "a", encoding="utf-8") as f:
        log_str = f"[{time.ctime()}] "
        for k, v in logs.items():
            log_str += f"{k}: {v:.6f}  "
        f.write(log_str.strip() + "\n")

# 训练时自动保存日志
trainer.add_callback(log_training_logs)
# ===============================================================

# ----------------------
# 开始训练
# ----------------------
trainer.train()

# 训练结束写入日志
with open(log_file, "a", encoding="utf-8") as f:
    f.write("\n" + "="*50 + "\n")
    f.write(f"训练完成时间：{time.ctime()}\n")
    f.write(f"最终平均 loss：{trainer.state.log_history[-1]['train_loss'] if 'train_loss' in trainer.state.log_history[-1] else 'N/A'}\n")
    f.write("训练成功完成！\n")
    f.write("="*50 + "\n")

# 保存
trainer.model.save_pretrained("../result/scitldr_qwen_lora")
tokenizer.save_pretrained("../result/scitldr_qwen_lora")

print("✅ 训练完成！模型已保存")
print(f"📄 训练日志已保存到：{log_file}")