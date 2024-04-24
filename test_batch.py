import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# 初始化Accelerator
accelerator = Accelerator()

checkpoint = "/home/lipz/BloomzLink/bloomz7b/bloomz-7b1"
# device = "cuda:0" if torch.cuda.is_available() else "cpu" 
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')  # 确保左侧填充
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# model.to_bettertransformer()
# model = model.to("cuda:0")

prompts = [
    "Translate to English: Je t’aime.",
    "Translate to English: 我爱你.",
    "Describe the plot of Inception.",
    "Describe the plot of The Friends."
]

# 设置最大长度，确保为8的倍数，使用左侧填充
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=128, pad_to_multiple_of=8)
# print(f"inputs:{inputs}\n")
model, inputs = accelerator.prepare(model, inputs)


# 生成输出
outputs = model.module.generate(**inputs, max_new_tokens=150)

# 解码并打印每个输出结果
# decoded_outputs = [tokenizer.batch_decode(output, skip_special_tokens=True) for output in outputs]
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(f"decoded_outputs :{decoded_outputs}\n")
accelerator.print("\nGenerated Texts:")
for i, text in enumerate(decoded_outputs):
    accelerator.print(f"Text {i+1}: {text}")
