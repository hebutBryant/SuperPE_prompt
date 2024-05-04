import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
checkpoint = "/home/lipz/BloomzLink/bloomz7b/bloomz-7b1"
# 初始化分词器和模型
tokenizer =  AutoTokenizer.from_pretrained(checkpoint,padding_side='left')
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# 准备批量数据，这里使用三个示例句子
texts = [
    "How can I improve my piano skills?"
]

# 对这些文本进行编码，添加必要的特殊符号
encoded_inputs = tokenizer(texts, padding='max_length', max_length=50, return_tensors="pt")


# 生成注意力掩码以忽略填充的影响
attention_mask = encoded_inputs['attention_mask']

# 使用模型的 generate 方法生成文本
# 这里设置 max_length 来限制生成文本的长度
outputs = model.generate(
    input_ids=encoded_inputs['input_ids'],
    attention_mask=attention_mask,
    max_new_tokens = 128,
    num_return_sequences=1
)
print(outputs)

# 解码并打印生成的文本
for i, output in enumerate(outputs):
    print(f"Generated text {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")



