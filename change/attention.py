#In this python files, I will achieve the attention mechamism in Superposition Prompt Paper. 
#我觉得它大体可以分为四个步骤，识别--拆分重组--计算--
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
def Identify(prompt):
    parts = {}

    # 修改正则表达式以更稳定地匹配部分
    instruction_match = re.search(r"###Instruction:(.*?)\n", prompt, re.S)
    question_match = re.search(r"###Question:(.*?)\s*$", prompt, re.S) 

    if instruction_match:
        parts['Instruction'] = instruction_match.group(1).strip()
    if question_match:
        parts['Question'] = question_match.group(1).strip()

    # 捕获所有的 Chunk
    chunk_matches = re.finditer(r"###(Chunk \d+):(.*?)\n", prompt, re.S)
    for match in chunk_matches:
        key = match.group(1).strip()
        value = match.group(2).strip()
        parts[key] = value

    return parts


def depart_and_combine(parts):
    # 提取基本部分
    instruction = parts.get('Instruction', '')
    question = parts.get('Question', '')
    
    prompts = []

    # 遍历所有 chunk，创建新的 prompt
    for key in parts:
        if key.startswith('Chunk'):
            prompt = f"###Instruction: {instruction}\n###Chunk: {parts[key]}\n###Question: {question}"
            prompts.append(prompt)
    
    return prompts

def Sparse_attention(model,tokenizer,prompts):

    return 



if __name__ == "__main__":
    template = [
        "###Instruction: Write a high-quality answer for the given question using only the following relevant search results.\n",
        "###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n",
        "###Chunk 2:Steve Jobs, along with Steve Wozniak, co-founded Apple in 1976, in Jobs' parents' garage. They revolutionized the tech industry by introducing the first Apple computer, which distinguished itself from others with a user-friendly graphical interface.\n",
        "###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n",
        "###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"
    ]
    template_str = ''.join(template)
    result = Identify(template_str)
    print(result)
    new_prompts = depart_and_combine(result)
    for prompt in new_prompts:
        print(prompt)
        print("----")
