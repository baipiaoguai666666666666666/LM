# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # ====================== 只需要改这里！======================
# # 你本地模型的文件夹路径（就是你刚才下载好的目录）
# model_path = "./qwen2.5-7B-Instruct"  
# # =================================================================

# # 加载本地分词器 + 模型
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map="auto",
#     trust_remote_code=True
# )

# # 测试提示词
# prompt = "请简单介绍一下大语言模型。"

# # 构造对话格式（Qwen 官方标准格式）
# messages = [
#     {"role": "user", "content": prompt}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )

# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # 生成回答
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512  # 不要用16384，7B模型会爆显存
# )

# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
# content = tokenizer.decode(output_ids, skip_special_tokens=True)

# print("\n===== 模型回答 =====")
# print("content:", content)


from peft import SFTConfig
import inspect

# 查看 SFTConfig 的签名
print(inspect.signature(SFTConfig.__init__))