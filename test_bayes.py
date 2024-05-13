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
# Logits 是神经网络最后一个线性层的输出，它们是未经归一化的原始预测值。在多类分类任务中，通常会对这些值应用 Softmax 函数来将它们转换为概率，这些概率表示模型对各个类别的预测置信度。
with torch.no_grad():
    #这里调用的是model的向前传播函数，不是generate函数
    outputs = model(input_ids=combined_ids)
    logits = outputs.logits


# 获取对应 continuation tokens 的 logits
# 注意：这里确保我们只取用于 continuation tokens 的 logits
shifted_logits = logits[:, input_ids.shape[1]-1:-1]  # 这里我们取从 input_ids 结束到倒数第二个 token 的 logits(看他们的注意力得分，再通过得分取得预测概率)
print(f"logits:{logits.shape}{logits}")
print(f"shifted_logits:{shifted_logits.shape}{shifted_logits}")
log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
print(f"continuation_ids{continuation_ids.shape}shifted_logits{shifted_logits.shape}")

# 检查对齐情况
assert continuation_ids.shape[1] == shifted_logits.shape[1], "Logits and continuation_ids length mismatch"

selected_log_probs = log_probs.gather(2, continuation_ids.unsqueeze(-1)).squeeze(-1)  # Select the log probabilities of the continuation tokens

# 计算整个生成文本的联合概率
joint_log_prob = selected_log_probs.sum()
joint_probability = torch.exp(joint_log_prob)
print(joint_probability)
print(f"Log probability of the continuation given the prefix: {joint_log_prob.item()}")
print(f"Probability of the continuation given the prefix: {joint_probability.item()}")
