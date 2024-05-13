# 在最后的自回归生成阶段，抛弃掉不相关的chunk
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def purning(k:int,instruction_logits:torch.Tensor,path_logits:torch.Tensor,actual_length:list,chunk_batch_ids:torch.Tensor,question_id:torch.Tensor):
    #首先计算在p条件下生成di的概率
    last_element = instruction_logits[0, -1, :]
    print(f"last_element shape{last_element.shape}")
    last_element = last_element.unsqueeze(0).unsqueeze(0) 
    #对path_logits分成path_num个 logits
    # split_tensor [path_num,1,seq_len,vocabulary]
    split_tensors = [path_logits[i:i+1] for i in range(path_logits.size(0))]
    extended_tensors = [torch.cat([last_element, tensor], dim=1) for tensor in split_tensors]
    final_tensors = [tensor[:, :-1, :] for tensor in extended_tensors]
    # for tensor in final_tensors:
    #     print(tensor.shape)
    log_probs = [torch.nn.functional.log_softmax(tensor, dim=-1) for tensor in final_tensors]
    sequence_probs = []
    query_probs = []
    _,query_len = question_id.shape
    for idx, log_prob in enumerate(log_probs):
        length = actual_length[idx]
        ids = chunk_batch_ids[idx, :length].unsqueeze(0).unsqueeze(-1)
        print(f"ids shape{ids.shape}")
        gathered_probs = torch.gather(log_prob, 2, ids)
        seq_prob = gathered_probs.sum()
        sequence_probs.append(seq_prob)
    print(f"sequence_probs{sequence_probs}")
    
    query_log_probs = [tensor[:,:-query_len,:] for tensor in log_probs]
    print(f"query_log_probs{query_log_probs}")
    question_id = question_id.unsqueeze(-1)
    for idx,log_prob in enumerate(query_log_probs):
        gathered_probs = torch.gather(log_prob, 2, question_id)
        seq_prob = gathered_probs.sum()
        query_probs.append(seq_prob)
    print(f"query_probs{query_probs}")
    #目前query_probs和sequence_probs里都是path_num个单元素张量
    combined_probs = [query_prob + seq_prob for query_prob, seq_prob in zip(query_probs, sequence_probs)]
    combined_probs_tensor = torch.tensor([prob.item() for prob in combined_probs])  # 转换为张量
    top_k_values, top_k_indices = torch.topk(combined_probs_tensor, k)
        
    print(f"Top {k} combined probabilities: {top_k_values}, at indices: {top_k_indices}")
    

    return top_k_indices


def path_cut(combined_kv:tuple,num_heads:int,top_k_indices:torch.Tensor):
    start_indices = top_k_indices*num_heads
    end_indices = start_indices + num_heads
    adjust_kv = []
    for i, (k, v) in enumerate(combined_kv):
        k_pieces = []
        v_pieces = []
        for start, end in zip(start_indices, end_indices):
            k_piece = k[start:end]
            v_piece = v[start:end]
            k_pieces.append(k_piece)
            v_pieces.append(v_piece)
        k_adjusted = torch.cat(k_pieces, dim=0)
        v_adjusted = torch.cat(v_pieces, dim=0)
        
        adjust_kv.append((k_adjusted, v_adjusted))



    return tuple(adjust_kv)


