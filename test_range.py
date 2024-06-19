import json
import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import re
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import time
torch.set_printoptions(threshold=1000000)

from change.Path_pruning import path_cut, purning, rank_past_key_values

checkpoint = 'Qwen/Qwen1.5-7B-Chat'
config = AutoConfig.from_pretrained(checkpoint)
weights_path = snapshot_download(checkpoint)
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)


tokenizer = AutoTokenizer.from_pretrained(checkpoint)


instruction_text = "Your instruction text here..."
context_text = "Your context text here..."
query_text = "Your query text here..."

# 将不同部分的文本转换为token并设置对应的范围
instruction_tokens = tokenizer(instruction_text, return_tensors="pt", padding="max_length", max_length=512).to("cuda")
context_tokens = tokenizer(context_text, return_tensors="pt", padding="max_length", max_length=1536).to("cuda") # 1536 = 2048 - 512
query_tokens = tokenizer(query_text, return_tensors="pt", padding="max_length", max_length=512).to("cuda") # 根据需要调整最大长度

# 将所有部分的token拼接在一起
input_ids = torch.cat([instruction_tokens["input_ids"], context_tokens["input_ids"], query_tokens["input_ids"]], dim=-1)
print(input_ids)

# 生成模型的输出
# outputs = model.generate(input_ids, max_new_tokens=512)
# generated_tokens = outputs[0]
# result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
# print(result)
