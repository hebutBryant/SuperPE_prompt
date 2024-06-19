import re
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator
from change.ArrangePositions import add_position,calculate_stride
from change.Path_pruning import purning,path_cut,rank_past_key_values
torch.set_printoptions(threshold=1000000)  # 可以根据你的张量大小调整这个值
num_heads = 32


if __name__ == "__main__":
    checkpoint = "/home/lipz/BloomzLink/bloomz7b/bloomz-7b1"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype="auto",device_map="auto")
    test_prompt = "Hello,how are you"

    #prefill 不加pad
    print("prefill 不加pad")
    inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[test_prompt], return_tensors="pt").to("cuda")
    inputs_ids = inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    print(f"input actual length:{inputs_ids.shape[1]}")
    attention_mask = inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    output = model.forward(input_ids=inputs_ids, use_cache=True, return_dict=True, output_hidden_states=True, attention_mask=attention_mask)


    #prefill 左填充加pad
    print("prefill 左填充加pad")
    inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[test_prompt],padding_strategy="max_length", max_length=6,return_tensors="pt").to("cuda")
    inputs_ids = inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    attention_mask = inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    output = model.forward(input_ids=inputs_ids, use_cache=True, return_dict=True, output_hidden_states=True, attention_mask=attention_mask)
    #prefill 右填充加pad
    print("prefill 右填充加pad")
    tokenizer2 = AutoTokenizer.from_pretrained(checkpoint,padding_side = 'right',padding=True,truncation=True)
    inputs = tokenizer2._batch_encode_plus(batch_text_or_text_pairs=[test_prompt],padding_strategy="max_length", max_length=6,return_tensors="pt").to("cuda")
    inputs_ids = inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    attention_mask = inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    output = model.forward(input_ids=inputs_ids, use_cache=True, return_dict=True, output_hidden_states=True, attention_mask=attention_mask)