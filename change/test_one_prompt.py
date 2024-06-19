import re
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator
from ArrangePositions import add_position,calculate_stride,calcuate_length
from Path_pruning import purning,path_cut,rank_past_key_values,purning2
import prompt
import process_RGB
torch.set_printoptions(threshold=1000000)  # 可以根据你的张量大小调整这个值
num_heads = 32
accelerator = Accelerator()



    # LONGEST = "longest"
    # MAX_LENGTH = "max_length"
    # DO_NOT_PAD = "do_not_pad"

# checkpoint = "meta-llama/Llama-2-13b-chat-hf"
checkpoint = "/home/lipz/BloomzLink/bloomz7b/bloomz-7b1"

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
    question = f"###Question: {question}\n"
    instruction = f"###Instruction: {instruction}\n"
    return instruction,prompts,question

def Sparse_attention(model, tokenizer, instruction, chunk_batch, question, max_length=110, top_k=2):
    instruction = [instruction]
    question = [question]
    instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=instruction, return_tensors="pt").to("cuda")
    chunk_batch_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=chunk_batch, padding_strategy="longest", return_tensors="pt").to("cuda")
    question_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=question, return_tensors="pt").to("cuda")
    instruction_ids = instruction_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_ids = chunk_batch_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    question_ids = question_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")

    instruction_attention_mask = instruction_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_attention_mask = chunk_batch_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    question_attention_mask = question_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    path_num,longest_length = chunk_batch_ids.shape


    expanded_question_ids = question_ids.expand(path_num, -1)
    combined_tensor = torch.cat((chunk_batch_ids, expanded_question_ids), dim=1)
    question_length = question_ids.shape[1]
    actual_length = calcuate_length(chunk_batch_ids)

    instruction_output = model.forward(input_ids=instruction_ids, use_cache=True, return_dict=True, output_hidden_states=True, attention_mask=instruction_attention_mask)
    instruction_logits = instruction_output.logits
    instruction_kv = instruction_output['past_key_values']
    expanded_past_key_values = tuple(
        (
            torch.repeat_interleave(layer[0], path_num, dim=0),
            torch.repeat_interleave(layer[1], path_num, dim=0)
        )
        for layer in instruction_kv
    )

    expanded_instruction_attention_mask = instruction_attention_mask.expand(path_num, -1)
    expanded_question_attention_mask = question_attention_mask.expand(path_num, -1)
    combined_attention_mask = torch.cat((expanded_instruction_attention_mask, chunk_batch_attention_mask, expanded_question_attention_mask), dim=1)
    combined_output = model.forward(input_ids=combined_tensor, use_cache=True, return_dict=True, output_hidden_states=True, past_key_values=expanded_past_key_values, attention_mask=combined_attention_mask)
    combined_kv = combined_output['past_key_values']
    combined_logits = combined_output.logits

    top_k_indices = purning(k=top_k, instruction_logits=instruction_logits, path_logits=combined_logits, chunk_batch_ids=chunk_batch_ids, question_id=question_ids, actual_length=actual_length)
    adjust_kv = path_cut(top_k_indices=top_k_indices, num_heads=num_heads, combined_kv=combined_kv)
    _, instruction_length = instruction_ids.shape
    final_past_key_values = rank_past_key_values(adjust_kv=adjust_kv, top_k=top_k, instruction_length=instruction_length)

    selected_tensor = combined_tensor[top_k_indices]
    flattened_tensor = selected_tensor.view(1, -1)
    combined_input_ids = torch.cat((instruction_ids, flattened_tensor), dim=1)

    selected_attention_mask = combined_attention_mask[top_k_indices]
    selected_attention_mask = selected_attention_mask[:, instruction_length:]
    flattened_attention_mask = selected_attention_mask.reshape(1, -1)
    final_attention_mask = torch.cat([instruction_attention_mask, flattened_attention_mask], dim=1)
    
    start_time = time.time()
    output = model.generate(inputs=combined_input_ids, past_key_values=final_past_key_values, use_cache=True, max_new_tokens=32, attention_mask=final_attention_mask)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model.generate execution time: {elapsed_time:.2f} seconds")
    print("Sparse attention:", tokenizer.decode(output[0]))

    inputs_length = combined_input_ids.shape[1]
    generated_tokens = output[0][inputs_length:]  # Get only the generated tokens
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print("Generated Text:", result)

    return result, elapsed_time






if __name__ == "__main__":
    template = [
        "###Instruction: Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk,no generalisations!\n",
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
    # print(process_RGB.prompts[0]["prompt"])
    # print("prompt",prompt.prompt1)



    template_str = ''.join(prompt.prompt_how)
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

    tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side = 'right',padding=True,truncation=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    # batch_result = Sparse_attention(model,tokenizer,instruction,chunk_batch,question)
    instruction = "Write a high-quality answer for the given question using only the following relevant search results,please answer in as much detail as possible based on chunk, no generalisations!"
    chunk1 = "During his tenure at Apple, Jobs was ousted from the company in 1985 but returned in 1997 to save the company from near bankruptcy. Under his leadership, Apple launched innovative products like the iPod, iPhone, and iPad."
    chunk2 = "In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs,which he later claimed profoundly influenced his creative strategies and business practices at Apple."
    question = "How did Steve Jobs' experiences and decisions shape the development and success of Apple?"

    try:
        template_str = ''.join(prompt.prompt_how)
        result = Identify(template_str)
        instruction, chunk_batch, question = depart_and_combine(result)

        result,  elapsed_time = Sparse_attention(model, tokenizer, instruction, chunk_batch, question)
        
        

    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory error encountered at iteration . Skipping to the next iteration.")
        torch.cuda.empty_cache()  # Clear the cache to free up memory
        
    except ValueError as ve:
        if 'Unable to create tensor' in str(ve):
            print(f"ValueError encountered at iteration : {ve}. Skipping to the next iteration.")
            
        else:
            raise ve  




    try:
        inputs = tokenizer.encode(prompt.prompt_how,
            return_tensors="pt"
        ).to("cuda")

        inputs_length = inputs.shape[1]
        print(f"inputs_length: {inputs_length}")

        start_time = time.time()
        outputs = model.generate(inputs, max_new_tokens=32)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model.generate execution time: {elapsed_time:.2f} seconds")

        print("Baseline:", tokenizer.decode(outputs[0]))

        generated_tokens = outputs[0][inputs_length:]  # Get only the generated tokens
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        print("Generated Text:", result)
        
        

    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory error encountered at iteration . Skipping to the next iteration.")
        torch.cuda.empty_cache()  # Clear the cache to free up memory
  

