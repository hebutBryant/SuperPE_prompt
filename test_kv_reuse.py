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

checkpoint = "bigscience/bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

instruction = "###Instruction: Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk,no generalisations!\n"

chunk = """###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.
###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.
"""

question = "###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"


###Response:


instruction = [instruction]
question = [question]
chunk = [chunk]
instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = instruction,return_tensors="pt").to("cuda")
chunk_batch_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = chunk,return_tensors="pt").to("cuda")
question_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = question,return_tensors="pt").to("cuda")
instruction_ids = instruction_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
chunk_batch_ids = chunk_batch_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
question_ids = question_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")

instruction_attention_mask = instruction_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
chunk_batch_attention_mask = chunk_batch_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
question_attention_mask = question_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")

combined_attention_mask = torch.cat(
    (instruction_attention_mask, chunk_batch_attention_mask, question_attention_mask),
    dim=1
)

print("--------------Reuse instruction kv----------------------------")
instruction_output = model.forward(input_ids = instruction_ids,use_cache = True,return_dict = True,output_hidden_states = True)
instruction_kv = instruction_output['past_key_values']
# print(f"chunk_ids:{chunk_batch_ids}")
# print(f"question_ids:{question_ids}")
chunk_question_combine_ids = torch.cat((instruction_ids,chunk_batch_ids,question_ids),dim=1)
print(f"chunk_question_combine_ids:{chunk_question_combine_ids}")
start_time = time.time()
output = model.generate(inputs = chunk_question_combine_ids,past_key_values = instruction_kv,use_cache = True,max_new_tokens = 128,attention_mask = combined_attention_mask)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model.generate execution time: {elapsed_time:.2f} seconds")
print("kv_reuse output :",tokenizer.decode(output[0]))


print("--------------Reuse instruction and Chunk kv----------------------------")
instruction_chunk_combine_ids = torch.cat((instruction_ids,chunk_batch_ids),dim=1)
print(f"instruction_chunk_combine_ids shape:{instruction_chunk_combine_ids.shape}")
instruction_chunk_combine_attention_mask = torch.cat((instruction_attention_mask,chunk_batch_attention_mask),dim=1)
print(f"instruction_chunk_combine_attention_mask shape:{instruction_chunk_combine_attention_mask.shape}")

chunk_output = model.forward(input_ids = chunk_batch_ids,past_key_values = instruction_kv,use_cache = True,attention_mask = instruction_chunk_combine_attention_mask)
instruction_chunk_combine_kv = chunk_output['past_key_values']
start_time = time.time()
output = model.generate(inputs = chunk_question_combine_ids,past_key_values = instruction_chunk_combine_kv,use_cache = True,max_new_tokens = 128,attention_mask = combined_attention_mask)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model.generate execution time: {elapsed_time:.2f} seconds")
print("kv_reuse output :",tokenizer.decode(output[0]))





print("------------------------Baseline--------------------------")
inputs = tokenizer.encode("###Instruction: Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk,no generalisations!\n###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n###Response:", return_tensors="pt").to("cuda")

start_time = time.time()
outputs = model.generate(inputs, max_new_tokens=256)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model.generate execution time: {elapsed_time:.2f} seconds")

print(tokenizer.decode(outputs[0]))