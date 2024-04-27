from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 初始化文本和生成第一部分
inputs = tokenizer.encode("Hello, how are you doing?", return_tensors="pt")
output_tokens = model.generate(inputs, max_length=10)
past_key_values = output_tokens.past_key_values

# 取最后一个 token 的 ID 继续生成
last_token_id = inputs[:, -1].unsqueeze(-1)

# 使用 past_key_values 继续生成
new_output = model(input_ids=last_token_id, past_key_values=past_key_values)
new_text = tokenizer.decode(new_output, skip_special_tokens=True)
print(new_text)
