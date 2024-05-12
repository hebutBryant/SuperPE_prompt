# In order to implement the positional allocation strategy proposed by the authors,
# we need to modify the Bloom model's forword function, here to do a simple implementation, a
# nd then apply the transformer package model folder Bloom_model under the folder
import sys
sys.path.append('/home/lipz/miniconda3/envs/myenv2/lib/python3.8/site-packages')

# 现在你可以导入你需要的模块
from transformers.models.bloom import modeling_bloom


import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ArrangePositions import calculate_stride

def forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        print("Sequence length:", fused_qkv.size(1))

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        print("Key layer shape:", key_layer.shape)
        if layer_past is not None:
            past_key, past_value = layer_past
            print("past_key shape:", past_key.shape)
            print("past_value shape:", past_value.shape)
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape
        # print("kv_length:",kv_length)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        # print("Query layer shape:", query_layer.shape)
        # print("Key layer shape:", key_layer.shape)
        # print(f"alibi:{alibi.shape}{alibi}")
        position_add_input = 
        
    





    return