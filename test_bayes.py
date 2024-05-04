from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 初始化分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()  # Set the model to evaluation mode

# 输入文本
prefix_text = "The quick brown fox"
continuation_text = "jumps over the lazy dog"

# 编码文本
input_ids = tokenizer.encode(prefix_text, return_tensors='pt')
continuation_ids = tokenizer.encode(continuation_text, return_tensors='pt')
print(f"input_ids:{input_ids}")
print(f"continuation_ids:{continuation_ids}")

# 组合前缀和生成文本的 token IDs
combined_ids = torch.cat((input_ids, continuation_ids), dim=-1)  # Remove the duplicated start token
print(f"combined_ids:{combined_ids}")
# 通过模型前向传递
with torch.no_grad():
    outputs = model(input_ids=combined_ids)
    logits = outputs.logits

# 获取对应 continuation tokens 的 logits
# 注意：这里确保我们只取用于 continuation tokens 的 logits
shifted_logits = logits[:, input_ids.shape[1]:-1]  # 这里我们取从 input_ids 结束到倒数第二个 token 的 logits
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

# 检查对齐情况
assert continuation_ids.shape[1] - 1 == shifted_logits.shape[1], "Logits and continuation_ids length mismatch"

selected_log_probs = log_probs.gather(2, continuation_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # Select the log probabilities of the continuation tokens

# 计算整个生成文本的联合概率
joint_log_prob = selected_log_probs.sum()
print(f"Log probability of the continuation given the prefix: {joint_log_prob.item()}")
