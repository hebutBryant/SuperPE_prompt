# 在最后的自回归生成阶段，抛弃掉不相关的chunk
import re
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator


def purning(k:int,threshold=0.5):