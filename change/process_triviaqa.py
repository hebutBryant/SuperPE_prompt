import json
import os
from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from ArrangePositions import add_position, calcuate_length,calculate_stride
from Path_pruning import purning,path_cut,rank_past_key_values,purning2
import re
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

checkpoint = 'Qwen/Qwen1.5-7B-Chat'
config = AutoConfig.from_pretrained(checkpoint)
weights_path = snapshot_download(checkpoint)
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path

def identify(prompt):
    parts = {}
    instruction_match = re.search(r"###Instruction:(.*?)\n", prompt, re.S)
    question_match = re.search(r"###Question:(.*?)\s*$", prompt, re.S)
    if instruction_match:
        parts['Instruction'] = instruction_match.group(1).strip()
    if question_match:
        parts['Question'] = question_match.group(1).strip()
    chunk_matches = re.finditer(r"###Chunk(\d+):(.*?)(?=\n###|$)", prompt, re.S)
    for match in chunk_matches:
        key = f"Chunk{match.group(1).strip()}"
        value = match.group(2).strip()
        parts[key] = value
    return parts

def depart_and_combine(parts):
    instruction = parts.get('Instruction', '')
    question = parts.get('Question', '')
    prompts = []
    for key in sorted(parts.keys()):
        if key.startswith('Chunk'):
            prompt = f"###Chunk {key[-1]}: {parts[key]}\n"
            prompts.append(prompt)
    question = f"###Question: {question}\n"
    instruction = f"###Instruction: {instruction}\n"
    return instruction, prompts, question

def list_to_string(lst):
    if lst and isinstance(lst, list):
        return "".join(lst)
    else:
        raise ValueError("Input must be a list containing string elements.")
# def Sparse_attention2(model, tokenizer, instruction, chunk_batch, question, max_length=1024, top_k=2):
#     instruction = [instruction]
#     question = [question]
#     instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=instruction, return_tensors="pt").to("cuda")
#     chunk_batch_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=chunk_batch, padding_strategy="max_length", max_length=max_length, return_tensors="pt").to("cuda")
#     question_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs=question, return_tensors="pt").to("cuda")
#     instruction_ids = instruction_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
#     chunk_batch_ids = chunk_batch_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")
#     question_ids = question_inputs.input_ids.clone().detach().to(dtype=torch.long).to("cuda")

#     instruction_attention_mask = instruction_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
#     chunk_batch_attention_mask = chunk_batch_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
#     question_attention_mask = question_inputs.attention_mask.clone().detach().to(dtype=torch.long).to("cuda")
#     path_num = chunk_batch_ids.shape[0]

#     expanded_question_ids = question_ids.expand(path_num, -1)
#     combined_tensor = torch.cat((chunk_batch_ids, expanded_question_ids), dim=1)
#     question_length = question_ids.shape[1]
#     actual_length = calcuate_length(combined_tensor)

#     instruction_output = model.forward(input_ids=instruction_ids, use_cache=True, return_dict=True, output_hidden_states=True, attention_mask=instruction_attention_mask)
#     instruction_logits = instruction_output.logits
#     instruction_kv = instruction_output['past_key_values']
#     expanded_past_key_values = tuple(
#         (
#             torch.repeat_interleave(layer[0], path_num, dim=0),
#             torch.repeat_interleave(layer[1], path_num, dim=0)
#         )
#         for layer in instruction_kv
#     )

#     expanded_instruction_attention_mask = instruction_attention_mask.expand(path_num, -1)
#     expanded_question_attention_mask = question_attention_mask.expand(path_num, -1)
#     combined_attention_mask = torch.cat((expanded_instruction_attention_mask, chunk_batch_attention_mask, expanded_question_attention_mask), dim=1)
#     combined_output = model.forward(input_ids=combined_tensor, use_cache=True, return_dict=True, output_hidden_states=True, past_key_values=expanded_past_key_values, attention_mask=combined_attention_mask)
#     combined_kv = combined_output['past_key_values']
#     combined_logits = combined_output.logits

#     top_k_indices = purning(k=top_k, instruction_logits=instruction_logits, path_logits=combined_logits, chunk_batch_ids=chunk_batch_ids, question_id=question_ids, actual_length=actual_length)
#     adjust_kv = path_cut(top_k_indices=top_k_indices, num_heads=num_heads, combined_kv=combined_kv)
#     _, instruction_length = instruction_ids.shape
#     final_past_key_values = rank_past_key_values(adjust_kv=adjust_kv, top_k=top_k, instruction_length=instruction_length)

#     selected_tensor = combined_tensor[top_k_indices]
#     flattened_tensor = selected_tensor.view(1, -1)
#     combined_input_ids = torch.cat((instruction_ids, flattened_tensor), dim=1)

#     selected_attention_mask = combined_attention_mask[top_k_indices]
#     selected_attention_mask = selected_attention_mask[:, instruction_length:]
#     flattened_attention_mask = selected_attention_mask.reshape(1, -1)
#     final_attention_mask = torch.cat([instruction_attention_mask, flattened_attention_mask], dim=1)
    
#     start_time = time.time()
#     output = model.generate(inputs=combined_input_ids, past_key_values=final_past_key_values, use_cache=True, max_new_tokens=32, attention_mask=final_attention_mask)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Model.generate execution time: {elapsed_time:.2f} seconds")
#     print("Sparse attention:", tokenizer.decode(output[0]))

#     inputs_length = combined_input_ids.shape[1]
#     generated_tokens = output[0][inputs_length:]  # Get only the generated tokens
#     result = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
#     print("Generated Text:", result)

#     return result,elapsed_time

def generate_prompt(instruction, chunks, question):
    chunk_text = ""
    for i, chunk in enumerate(chunks, 1):
        chunk_text += f"###Chunk{i}: {chunk}\n"
    prompt = (
        f"###Instruction: {instruction}\n"
        f"{chunk_text}"
        f"###Question: {question}\n"
    )
    return prompt

def is_correct_qwen(answer, generate_token):
    for ans in answer:
        if ans in generate_token:
            return True
    return False

with open('data/triviaqa_link/qa/verified-wikipedia-dev.json', 'r') as file:
    data = json.load(file)

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"Content not found for {filepath}"
    
if __name__ == "__main__":

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
            question=question
        )
        prompts.append(prompt)
        prompts_answer.append(combined_aliases)

    for prompt in prompts[:5]:
        print(prompt)
        print("\n" + "="*80 + "\n")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-7B-Chat",
        torch_dtype="auto",
        device_map="auto"
    )

    def to_device(batch, device):
        if isinstance(batch, (tuple, list)):
            return [to_device(t, device) for t in batch]
        return batch.to(device)

    correct_count = 0

    for i, prompt in enumerate(prompts):
        try:
            result = identify(prompt)
            instruction, chunk_batch, question = depart_and_combine(result)

            messages = [
                {"role": "system", "content": instruction},
                {"role": "document", "content": list_to_string(chunk_batch)},
                {"role": "user", "content": question}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt", padding="max_length", max_length=50).to("cuda:0")
            outputs = model.generate(model_inputs["input_ids"], max_new_tokens=512)
            input_length = model_inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(f"Generated Text for prompt {i+1}:", result)

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

    # for i, prompt in enumerate(prompts):
    #     try:
    #         result = identify(prompt)
    #         instruction, chunk_batch, question = depart_and_combine(result)

    #         messages = [
    #             {"role": "system", "content": instruction},
    #             {"role": "document", "content": list_to_string(chunk_batch)},
    #             {"role": "user", "content": question}
    #         ]

    #         text = tokenizer.apply_chat_template(
    #             messages,
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )

    #         model_inputs = tokenizer([text], return_tensors="pt", padding="max_length", max_length=50).to("cuda:0")
    #         outputs = model.generate(model_inputs["input_ids"], max_new_tokens=512)
    #         input_length = model_inputs["input_ids"].shape[1]
    #         generated_tokens = outputs[0][input_length:]
    #         result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    #         print(f"Generated Text for prompt {i+1}:", result)

    #         if is_correct_qwen(answer=prompts_answer[i], generate_token=result):
    #             correct_count += 1

    #     except torch.cuda.OutOfMemoryError as e:
    #         print(f"CUDA out of memory error on prompt {i+1}: {e}")
    #         torch.cuda.empty_cache()
    #         continue
    #     except RuntimeError as e:
    #         print(f"Runtime error on prompt {i+1}: {e}")
    #         torch.cuda.empty_cache()
    #         continue

    # print(f"Total correct answers: {correct_count} out of {len(prompts)}")
