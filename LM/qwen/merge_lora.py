from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径
base_model_path = "qwen2.5-7B-Instruct"
lora_path = "result/demo"
output_path ="qwen2.5-7B-finetuned-merged"

# 加载
tokenizer = AutoTokenizer.from_pretrained(
     base_model_path,
     trust_remote_code=True,
     local_files_only=True
)
base_model = AutoModelForCausalLM.from_pretrained(
     base_model_path,
     device_map="auto",
     torch_dtype="auto",
     trust_remote_code=True,
     local_files_only=True
)

# 加载 LoRA 并合并
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# 保存完整模型
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"✅ 合并完成！模型保存到：{output_path}")