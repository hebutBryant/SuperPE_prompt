import torch
from transformers import GPT2LMHeadModel, GPT2Config

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )

# 加载配置和模型
config = GPT2Config.from_pretrained('gpt2')
model = CustomGPT2LMHeadModel.from_pretrained('gpt2', config=config)

# 假设 input_ids 已经准备好
input_ids = torch.tensor([[50256, 257, 262, 2062, 262, 393]])

# 创建三角形掩码
seq_length = input_ids.size(1)
triangular_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.long))

# 使用自定义掩码进行生成
outputs = model.generate(input_ids, attention_mask=triangular_mask)
