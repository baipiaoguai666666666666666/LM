1.需要科学上网下载模型，或者手动下载之后上传亦可
2.下载模型之后，将数据集下载，数据集为SCItldr数据集
3.将数据集格式转换成可用的message格式，jsonl转成message格式脚本为jsonl_to_message.py
4. 运行fine-tuning.py

二、部署（vLLM）
pip install vllm -U
pip install "vllm[openai]" -U
pip install peft transformers accelerate -U

2.运行 merge_lora.py
步骤 3：vLLM 启动 API 服务
python -m vllm.entrypoints.openai.api_server \
--model ./qwen2.5-7B-sql-merged \
--trust-remote-code \
--host 0.0.0.0 \
--port 8866 \
--dtype auto \
--gpu-memory-utilization 0.9 \
--served-model-name sql-qwen

3.出现下面内容代表成功
Uvicorn running on http://0.0.0.0:8866
