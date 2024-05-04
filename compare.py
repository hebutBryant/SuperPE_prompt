import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator

def generate_answer(model_name, question, max_length=50):
    # 加载预训练模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 将问题编码为模型能够理解的格式
    inputs = tokenizer.encode(question, return_tensors="pt")

    # 生成答案
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)

    # 解码生成的答案
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 模型名和问题
model_name = "facebook/llama-2-7b"
question = "What is the capital of France?"

# 生成答案
answer = generate_answer(model_name, question)
print(f"Answer: {answer}")