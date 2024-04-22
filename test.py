# /home/lipz/BloomzLink/bloomz3b/bloomz-3b  权重路径
from transformers import BloomModel, BloomTokenizerFast

# 设置模型和权重的路径
model_path = "/home/lipz/BloomzLink/bloomz3b/bloomz-3b/pytorch_model.bin "

# 加载预训练模型和对应的tokenizer
model = BloomModel.from_pretrained(model_path)
tokenizer = BloomTokenizerFast.from_pretrained(model_path)

# 函数用于生成文本
def generate_text(prompt, max_length=50):
    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成输出文本的token
    output_tokens = model.generate(**inputs, max_length=max_length)

    # 解码生成的token以获取文本
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return generated_text

# 使用模型
prompt = "今天的天气如何？"
generated_answer = generate_text(prompt)

print(generated_answer)
