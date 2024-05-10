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
    





    return