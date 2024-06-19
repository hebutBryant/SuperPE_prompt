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

# checkpoint = "bigscience/bloomz-7b1"
checkpoint = "Qwen/Qwen1.5-4B-Chat"

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


#加入模板
print("------------------------Baseline2--------------------------")
prompt = "###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"
messages = [
    {"role": "system", "content": "Write a high-quality answer for the given question using only the following relevant search results"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f"template:{text}")

model_inputs = tokenizer([text], return_tensors="pt",padding = "max_length",max_length=50).to("cuda")
print(f"model_inputs{model_inputs}")
outputs = model.generate(model_inputs["input_ids"], max_new_tokens=512)
generated_tokens = outputs[0] # Get only the generated tokens
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Generated Text:", result)


#input_ids长度大于past_key_values的复用
print("--------------All kv reuse----------------------------")
# Assuming you have already defined your tokenizer and model
# Define the prompt
prompt = "###Instruction: Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk,no generalisations!\n###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"

# Encode the prompt
inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[prompt], return_tensors="pt").to("cuda")
input_ids = inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
attention_mask = inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")

# Forward pass to get the hidden state and past key values
hidden_state = model.forward(input_ids=input_ids, past_key_values=None, use_cache=True, attention_mask=attention_mask)
past_key_value = hidden_state['past_key_values']
logits = hidden_state.logits

# Get the highest probability token's id for the last generated token
next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

# Append the token id to the input_ids
output_ids = torch.cat([input_ids, next_token_id], dim=-1)

attention_mask = torch.cat([attention_mask,torch.tensor([[1]],dtype=int,device="cuda")],dim=-1)

print(f"attention mask shape:{attention_mask.shape}")
generated_text1 = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
print("generated_text1:",generated_text1)
print(output_ids.shape)
for i, layer in enumerate(past_key_value):
    print(f"Layer {i} key shape: {layer[0].shape}, value shape: {layer[1].shape}")
    break
output = model.generate(inputs = output_ids ,past_key_values = past_key_value,use_cache = True,max_new_tokens = 128,attention_mask = attention_mask)


# Decode the generated output ids to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

#input_ids长度等于past_key_values的复用
# print("--------------All kv reuse2----------------------------")
# prompt = "###Instruction: Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk,no generalisations!\n###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"

# # Encode the prompt
# inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[prompt], return_tensors="pt").to("cuda")
# input_ids = inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
# attention_mask = inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")

# # Forward pass to get the hidden state and past key values
# hidden_state = model.forward(input_ids=input_ids, past_key_values=None, use_cache=True, attention_mask=attention_mask)
# past_key_value = hidden_state['past_key_values']
# logits = hidden_state.logits

# for i, layer in enumerate(past_key_value):
#     print(f"Layer {i} key shape: {layer[0].shape}, value shape: {layer[1].shape}")
#     break
# output = model.generate(inputs = input_ids ,past_key_values = past_key_value,use_cache = True,max_new_tokens = 128,attention_mask = attention_mask)


# # Decode the generated output ids to text
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)


print("--------------All kv reuse qwen1.5 and superposition version----------------------------")

prompt = "###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n"
messages = [
    {"role": "system", "content": "Write a high-quality answer for the given question using only the following relevant search results"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

suffix = "###Response"

inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[text], return_tensors="pt").to("cuda")
suffix_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = [suffix],return_tensors="pt").to("cuda")
model_input_ids = model_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
model_input_mask = model_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
suffix_input_ids = suffix_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
suffix_input_mask = suffix_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
combine_ids = torch.cat((model_input_ids,suffix_input_ids),dim=1)
combine_mask = torch.cat((model_input_mask,suffix_input_mask),dim=1)
# Forward pass to get past key values
model_output = model.forward(input_ids=model_input_ids, use_cache=True, return_dict=True, attention_mask=model_input_mask)
past_key_values = model_output.past_key_values

outputs = model.generate(
    input_ids=combine_ids,
    attention_mask=combine_mask,
    past_key_values=past_key_values,
    use_cache=True,
    max_new_tokens=512,
)

# Decode the generated tokens
generated_tokens = outputs[0]
result = tokenizer.decode(generated_tokens, skip_special_tokens=False)
print("Generated Text:", result)




