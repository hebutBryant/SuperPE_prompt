#In this python files, I will achieve the attention mechamism in Superposition Prompt Paper. 
#我觉得它大体可以分为四个步骤，识别--拆分重组--计算--
import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


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
            prompt = f"###Chunk: {parts[key]}\n###Question: {question}"
            prompts.append(prompt)
    
    return instruction,prompts



# class PaddingStrategy(ExplicitEnum):
#     """
#     Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
#     IDE.
#     """
# 三种填充策略
#     LONGEST = "longest"
#     MAX_LENGTH = "max_length"
#     DO_NOT_PAD = "do_not_pad"


# hidden_state 就是隐藏层（多头注意力加FNN）的输出
# return BaseModelOutputWithPastAndCrossAttentions(
#     last_hidden_state=hidden_states,
#     past_key_values=presents,
#     hidden_states=all_hidden_states,
#     attentions=all_self_attentions,
# )
# Shape of hidden state at layer 0: torch.Size([3, 71, 4096])
# 第一个数字代表 batch size，即处理的样本数量。
# 第二个数字表示序列长度，即输入中的token数量。
# 第三个数字代表隐藏层的宽度（即每个token的特征维度）
# Layer 1 Keys Shape: torch.Size([32, 128, 17])
# Layer 1 Values Shape: torch.Size([32, 17, 128])
# Layer 1 Keys Shape: torch.Size([96, 128, 71])
# Layer 1 Values Shape: torch.Size([96, 71, 128])
#first dim: batch_size*head_num , second dim(17): token num , third dim(128): keys or values dim           we can see 4096(hidden_dim) = 32(head_num)*128(kv_dim)

def Sparse_attention(model,tokenizer,instruction,chunk_batch,max_length=128):
    instruction = [instruction]
    instruction_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = instruction)
    chunk_batch_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = chunk_batch, padding_strategy = "maxlength",max_length = max_length)
    #path全部编码完成
    instruction_ids = torch.tensor(instruction_inputs.input_ids,dtype=torch.long)
    instruction_attention_mask = torch.tensor(instruction_inputs.attention_mask , dtype=torch.long)
    chunk_batch_ids = torch.tensor(chunk_batch_inputs.attention_mask , dtype=torch.long)
    chunk_batch_attention_mask = torch.tensor(chunk_batch_inputs.attention_mask , dtype=torch.long)
    #首先对instruction 在模型编码器中进行前向传播
    # instruction_output 将包含一个名为 past_key_values 的元素，它包含了模型所有层的键值对缓存。
    instruction_output = model.forward(input_ids = instruction_ids,attention_mask = instruction_attention_mask,use_cache = True,return_dict = True,output_hidden_states = True)
    #拿到instruction 编码所计算得到的KV Cache
    instruction_hidden_states = instruction_output['hidden_states']
    # for idx, hidden_state in enumerate(instruction_hidden_states):
    #     print(f"Shape of hidden state at layer {idx}: {hidden_state.shape}")
    instruction_kv = instruction_output['past_key_values']
    print("-------------------------------------------\n")
    #如果不扩展past_key_values，在forward中cat past_key和key_layer时 会发生维度不匹配
    expanded_past_key_values = tuple(
    (
        torch.repeat_interleave(layer[0], 3, dim=0),
        torch.repeat_interleave(layer[1], 3, dim=0)
    )
    for layer in instruction_kv
    )
    #传入instruction的past_key_value 会造成attention_mask维度不匹配的报错，因为batch中每个序列的attention_mask需要向前扩展len(instruction_token)数个掩码长度
    #简单的做法就是把instruction_attention_mask直接拼到 batch中每个mask前面
    # print(chunk_batch_attention_mask)
    instruction_attention_mask_expand = instruction_attention_mask.repeat(chunk_batch_attention_mask.size(0),1)
    cat_attention_mask = torch.cat((instruction_attention_mask_expand,chunk_batch_attention_mask),dim=1)
    # print(cat_attention_mask)

    chunk_batch_output = model.forward(input_ids = chunk_batch_ids,attention_mask = cat_attention_mask,use_cache = True,return_dict = True,output_hidden_states = True,past_key_values = expanded_past_key_values)
    chunk_batch_hidden_states = chunk_batch_output['hidden_states']
    #region
    # for idx, hidden_state in enumerate(chunk_batch_hidden_states):
    #     print(f"Shape of hidden state at layer {idx}: {hidden_state.shape}")
    # # instruction_hidden_states_shape = instruction_hidden_states.shape()
    # output = model.forward(input_ids = input_ids,attention_mask = attention_mask,use_cache = True,return_dict = True)
    # print("-------------------------------------------\n")
    # # for layer_idx, (keys, values) in enumerate(instruction_kv):
    # #     print(f"Layer {layer_idx+1} Keys Shape: {keys.shape}")
    # #     print(f"Layer {layer_idx+1} Values Shape: {values.shape}")
    # print("-------------------------------------------\n")
        #endregion
    chunk_batch_kv = chunk_batch_output['past_key_values']

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
    # print(result)
    instruction,chunk_batch= depart_and_combine(result)
    print("instruction:")
    print(instruction)
    print("------------------\n")
    print("chunk_batch:")
    print(chunk_batch)
    print("------------------\n")

    # for prompt in new_prompts:
    #     print(prompt)
    #     print("----")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')  # 确保左侧填充
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    batch_result = Sparse_attention(model,tokenizer,instruction,chunk_batch)
    # logging.info("attention shows")
    print(batch_result)
    # lt =  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # print("length lt",len(lt))

