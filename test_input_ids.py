# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # 初始化文本和生成第一部分
# inputs = tokenizer.encode("Hello, how are you doing?", return_tensors="pt")
# output_tokens = model.generate(inputs, max_length=50)
# past_key_values = output_tokens.past_key_values

# # 取最后一个 token 的 ID 继续生成
# last_token_id = inputs[:, -1].unsqueeze(-1)

# # 使用 past_key_values 继续生成
# new_output = model(input_ids=last_token_id, past_key_values=past_key_values)
# new_text = tokenizer.decode(new_output, skip_special_tokens=True)
# print(new_text)


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 初始化模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 输入文本
input_text = "The science of today is the technology of tomorrow."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 第一次生成，没有使用 past_key_values
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
print("First generation without past_key_values:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 存储 past_key_values
with torch.no_grad():
    output, past_key_values = model(input_ids, use_cache=True, return_dict=True).values()

# 使用已计算的 past_key_values 进行第二次生成
input_ids_new = tokenizer.encode(" Technology is evolving", return_tensors="pt")
outputs = model.generate(input_ids_new, max_length=70, num_return_sequences=1, use_cache=True, past_key_values=past_key_values)
print("\nSecond generation using past_key_values:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
