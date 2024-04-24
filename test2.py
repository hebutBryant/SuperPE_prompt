import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "/home/lipz/BloomzLink/bloomz3b/bloomz-3b"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


print("Tokenizer Configuration:")
print(tokenizer.name_or_path)  # 显示分词器的路径或名称
print(tokenizer.model_max_length)  # 显示模型允许的最大输入长度

print("\nModel Configuration:")
print(model.config)  # 打印模型的全部配置信息

# 准备输入数据
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")

# 生成输出
outputs = model.generate(inputs)

# 解码并打印输出结果
print("\nGenerated Text:")
print(tokenizer.decode(outputs[0]))
