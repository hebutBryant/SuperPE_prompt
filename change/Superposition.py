import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator
from ArrangePositions import add_position,calculate_stride
from Path_pruning import purning,path_cut,rank_past_key_values
torch.set_printoptions(threshold=1000000)  # 可以根据你的张量大小调整这个值
num_heads = 32


checkpoint = "bigscience/bloomz-7b1"

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
            prompt = f"###Chunk: {parts[key]}\n"
            prompts.append(prompt)
    
    return instruction,prompts,question

def Sparse_attention(model,tokenizer,instruction,chunk_batch,question,max_length=64,top_k = 2):
    instruction = [instruction]
    question = [question]
    instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = instruction,return_tensors="pt").to("cuda")
    chunk_batch_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = chunk_batch, padding_strategy = "max_length",max_length = max_length,return_tensors="pt").to("cuda")
    question_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = question,return_tensors="pt").to("cuda")
    instruction_ids = instruction_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_ids = chunk_batch_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    question_ids = question_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")

    instruction_attention_mask = instruction_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_attention_mask = chunk_batch_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    question_attention_mask = question_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")

    # print(f"attention mask shape:{instruction_attention_mask.shape}{chunk_batch_attention_mask.shape}{question_attention_mask.shape}")
    # print(f"instruction_ids:{instruction_ids}")
    # print(f"{instruction_ids.shape}--{chunk_batch_ids.shape}--{question_ids.shape}")
    expanded_question_ids = question_ids.expand(3, -1)  # -1 表示这一维度保持不变
    # 拼接 chunk_batch_ids 和 expanded_question_ids 沿着第二维
    combined_tensor = torch.cat((chunk_batch_ids, expanded_question_ids), dim=1)
    # print(f"Combined tensor shape: {combined_tensor.shape}")
    # print(combined_tensor)
    _,actual_length = calculate_stride(combined_tensor)
    # print(f"actual_length{actual_length}")

    # instruction_ids,chunk_batch_ids = add_position(instruction_ids,combined_tensor)

    instruction_output = model.forward(input_ids = instruction_ids,use_cache = True,return_dict = True,output_hidden_states = True)
    instruction_logits = instruction_output.logits
    # print(f"instruction_logits  shape:{instruction_logits.shape}")
    instruction_kv = instruction_output['past_key_values']
    # print(f"instruction_kv:{instruction_kv}")
    expanded_past_key_values = tuple(
    (
        torch.repeat_interleave(layer[0], 3, dim=0),
        torch.repeat_interleave(layer[1], 3, dim=0)
    )
    for layer in instruction_kv
    )
    expanded_question_attention_mask = question_attention_mask.expand(3, -1)
    combined_attention_mask = torch.cat((chunk_batch_attention_mask, expanded_question_attention_mask), dim=1)
    combined_output = model.forward(input_ids = combined_tensor,use_cache = True,return_dict = True,output_hidden_states = True,past_key_values = expanded_past_key_values)
    combined_kv = combined_output['past_key_values']
    # for i, (k, v) in enumerate(combined_kv):
    #     print(f"Layer {i} - Key shape: {k.shape}, Value shape: {v.shape}")
    combined_logits = combined_output.logits
    # print(f"combined_output logits shape:{combined_logits .shape}")
    # print(f"combined_kv{combined_kv}")

    top_k_indices = purning(k=top_k,instruction_logits=instruction_logits,path_logits=combined_logits,chunk_batch_ids=chunk_batch_ids,question_id=question_ids,actual_length=actual_length)
    #拿到top_k_indices后的第一步 把 [batch_size * self.num_heads, self.head_dim, q_length]的第一个维度改为 k*self.num_heads
    adjust_kv = path_cut(top_k_indices=top_k_indices,num_heads=num_heads,combined_kv=combined_kv)
    # for i, (k, v) in enumerate(adjust_kv):
    #     print(f"Layer {i} - Key shape: {k.shape}, Value shape: {v.shape}")
    
    #对于past_key_values 是含有layer_num个element的元组，每一个元组又包含两个tensor k v
    #5/13 第二步变换 [k*self.num_heads, self.head_dim, q_length] 变为 [num_heads,head_dim,q_length+length(di+q)]
    #as for model.generate function, we need a input_id , I use <bos> token_id = 1
    #第二种 first_input_id 思路 获取top_k_indices 表示的query 的logits
    first_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = ['### Response:'],return_tensors="pt").to("cuda")
    first_input_id = first_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    # print(f"first_input_id{first_input_id}")
    _,instruction_length = instruction_ids.shape
    final_past_key_values = rank_past_key_values(adjust_kv=adjust_kv,top_k=top_k,instruction_length=instruction_length)
    # Layer 29 - Key shape: torch.Size([32, 128, 177]), Value shape: torch.Size([32, 177, 128])
    # for i, (k, v) in enumerate(final_past_key_values):
    #     print(f"Layer {i} - Key shape: {k.shape}, Value shape: {v.shape}")
   
    
    selected_attention_mask = combined_attention_mask[top_k_indices]
    flattened_attention_mask = selected_attention_mask.reshape(1, -1)
    final_attention_mask = torch.cat([instruction_attention_mask, flattened_attention_mask], dim=1)
    # print("Final Attention Mask shape:", final_attention_mask)  # 应为 [1, 177]

    output = model.generate(inputs = first_input_id,past_key_values = final_past_key_values,use_cache = True,max_new_tokens = 128,attention_mask = final_attention_mask)
    print("Sparse attention :",tokenizer.decode(output[0]))



    return






if __name__ == "__main__":
    template = [
        "###Instruction: Write a high-quality answer for the given question using only the following relevant search results.\n",
        "###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n",
        "###Chunk 2:Steve Jobs, along with Steve Wozniak, co-founded Apple in 1976, in Jobs' parents' garage. They revolutionized the tech industry by introducing the first Apple computer, which distinguished itself from others with a user-friendly graphical interface.\n",
        "###Chunk 3:During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n",
        "###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?"
    ]
    template2 = [
        "###Instruction: Write a high-quality answer for the given question using only the following relevant search results.\n",
        "###Chunk :In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n",
        "###Chunk :During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n",
        "###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?"
    ]


    template_str = ''.join(template)
    result = Identify(template_str)
    # print(result)
    instruction,chunk_batch,question= depart_and_combine(result)
    print("instruction:")
    print(instruction)
    print("------------------\n")
    print("chunk_batch:")
    print(chunk_batch)
    print("------------------\n")
    print(question)
    print("------------------\n")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side = 'right')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    batch_result = Sparse_attention(model,tokenizer,instruction,chunk_batch,question)
    inputs = tokenizer.encode("###Instruction: Write a high-quality answer for the given question using only the following relevant search results.\n",
        "###Chunk :In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Chunk :During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n### Response:", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs,max_new_tokens = 128)
    print("Baseline:",tokenizer.decode(outputs[0]))