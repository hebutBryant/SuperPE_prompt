import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ArrangePositions import add_position, calcuate_length,calculate_stride
from Path_pruning import rank_past_key_values
from process_triviaqa import is_correct_qwen,read_file,identify,depart_and_combine,is_correct_qwen,list_to_string,generate_prompt
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

start = '<|im_start|>'
end = '<|im_end|>'
num_heads = 32
suffix = "<|im_end|>\n<|im_start|>assistant\n###Response:"
# 使用 cache_path 作为路径进行后续操作

def Identify_and_depart(text):
    parts = {}

    # Extract the 'system' part with 'system' and 'user' labels included
    system_match = re.search(r"(system\n.*?user\n)", text, re.S)
    if system_match:
        parts['system'] = start + system_match.group(0).strip()+ "\n"

    # Extract the 'user' part and chunks
    chunk_matches = re.finditer(r"###Chunk (\d+):(.*?)(?=\n###Chunk|\n###Question|$)", text, re.S)
    chunks = [f"###Chunk {match.group(1)}:{match.group(2).strip()}" for match in chunk_matches]
    parts['chunk_list'] = chunks

    # Extract the question with newline character at the end
    question_match = re.search(r"(###Question:.*?)(?=\n|$)", text, re.S)
    if question_match:
        parts['question'] =  question_match.group(1) + "\n"

    instruction = parts['system']
    question = parts['question']
    chunk_batch = parts['chunk_list']

    return instruction,chunk_batch,question

def Sparse_attention(model, tokenizer, instruction, chunk_batch, question,suffix = suffix,max_length=4096, top_k=2):
    instruction = [instruction]
    question = [question]
    suffix = [suffix]
    instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=instruction, return_tensors="pt").to("cuda")
    chunk_batch_inputs = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=chunk_batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to("cuda")
    question_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=question, return_tensors="pt").to("cuda")
    suffix_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=suffix, return_tensors="pt").to("cuda")
    instruction_ids = instruction_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_ids = chunk_batch_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    question_ids = question_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
    suffix_ids = suffix_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")

    instruction_attention_mask = instruction_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    chunk_batch_attention_mask = chunk_batch_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    question_attention_mask = question_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    suffix_attention_mask =  suffix_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
    path_num = chunk_batch_ids.shape[0]

    expanded_question_ids = question_ids.expand(path_num, -1)
    combined_tensor = torch.cat((chunk_batch_ids, expanded_question_ids), dim=1)
    question_length = question_ids.shape[1]
    actual_length = calcuate_length(combined_tensor)

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
    print(f"instruction length:{instruction_ids.shape[1]},question length:{question_ids.shape[1]}")
    print("-------combined_kv---------")
    for i, layer in enumerate(combined_kv):
        print(f"Layer {i} key shape: {layer[0].shape}, value shape: {layer[1].shape}")


    _, instruction_length = instruction_ids.shape


    final_past_key_values = rank_past_key_values(adjust_kv=combined_kv, top_k=path_num, instruction_length=instruction_length)
    # selected_tensor = combined_tensor[top_k_indices]
    flattened_tensor = combined_tensor.view(1, -1)
    combined_input_ids = torch.cat((instruction_ids, flattened_tensor), dim=1)

    # selected_attention_mask = combined_attention_mask[top_k_indices]
    selected_attention_mask = combined_attention_mask[:, instruction_length:]
    flattened_attention_mask = selected_attention_mask.reshape(1, -1)
    final_attention_mask = torch.cat([instruction_attention_mask, flattened_attention_mask], dim=1)


    #concat ###Response去激活generate
    combined_input_ids = torch.cat([combined_input_ids,suffix_ids],dim=-1)
    final_attention_mask = torch.cat([final_attention_mask,suffix_attention_mask],dim=-1)
    print(f"{combined_input_ids.shape},{final_attention_mask.shape}")
    print("-------final_past_key_values---------")
    for i, layer in enumerate(final_past_key_values):
        print(f"Layer {i} key shape: {layer[0].shape}, value shape: {layer[1].shape}")
    
    start_time = time.time()
    output = model.generate(inputs=combined_input_ids, past_key_values=final_past_key_values, use_cache=True, max_new_tokens=512, attention_mask=final_attention_mask)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model.generate execution time: {elapsed_time:.2f} seconds")
    print("Sparse attention:", tokenizer.decode(output[0]))

    inputs_length = combined_input_ids.shape[1]
    generated_tokens = output[0][inputs_length:]  # Get only the generated tokens
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print("Generated Text:", result)

    return result, elapsed_time


def generate_prompt_qwen(instruction:str,chunks:list,question:str,tokenizer):
    chunk_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunk_text += f"###Chunk{i}: {chunk}\n"
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": chunk_text+question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text





device = "cuda" if torch.cuda.is_available() else "cpu" # the device to load the model onto
if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
    inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=[suffix], return_tensors="pt").to("cuda")
    print("inputs",tokenizer.decode(inputs.input_ids[0]))
    


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
    print(text)



    instruction,chunk_batch,question = Identify_and_depart(text=text)
    print("instruction:")
    print(instruction)
    print("------------------\n")
    print("chunk_batch:")
    print(chunk_batch)
    print("------------------\n")
    print(question)
    print("------------------\n")

    # result, elapsed_time = Sparse_attention(model, tokenizer, instruction, chunk_batch, question)

    with open('data/triviaqa_link/qa/verified-wikipedia-dev.json', 'r') as file:
        data = json.load(file)

    prompts = []
    prompts_answer = []
    for item in data['Data']:
        question = item['Question']
        chunks = []
        for entity_page in item['EntityPages']:
            filename = entity_page['Filename']
            filepath = os.path.join('data/triviaqa_link/evidence/wikipedia', filename)
            chunk_content = read_file(filepath)
            chunks.append(chunk_content)
            aliases = item['Answer']['Aliases']
            normalized_aliases = item['Answer']['NormalizedAliases']
            combined_aliases = aliases + normalized_aliases
        prompt = generate_prompt(
            instruction="Write a high-quality answer for the given question using only the following relevant search results",
            chunks=chunks,
            question=question,
        )
        prompts.append(prompt)
        prompts_answer.append(combined_aliases)

    correct_count = 0
    for i, prompt in enumerate(prompts):

        try:
            result = identify(prompt)
            instruction, chunk_batch, question = depart_and_combine(result)
            text = generate_prompt_qwen(instruction = instruction,chunks=chunk_batch,question=question,tokenizer=tokenizer)
            print(f"text:{text}")
            instruction,chunk_batch,question = Identify_and_depart(text=text)
            result, elapsed_time = Sparse_attention(model, tokenizer, instruction, chunk_batch, question)
            if is_correct_qwen(answer=prompts_answer[i], generate_token=result):
                correct_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error on prompt {i+1}: {e}")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as e:
            print(f"Runtime error on prompt {i+1}: {e}")
            torch.cuda.empty_cache()
            continue
    print(f"Total correct answers: {correct_count} out of {len(prompts)}")


