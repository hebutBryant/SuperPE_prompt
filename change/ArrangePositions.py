import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator


def calculate_stride(path:torch.Tensor):
    path_num,max_length= path.shape
    mean = 0.0
    reciprocal_sum = 0.0

    actual_length = [0]*path_num
    stride_length = [0]*path_num
    for i in range(path_num):
        length = 0
        for j in range(max_length):
            if path[i][j] != 3:
                length = length+1
            else:
                actual_length[i] = length
                reciprocal_sum = reciprocal_sum + 1/length
                break
    
    mean = path_num/reciprocal_sum
    for i in range(path_num):
        stride_length[i] = mean/actual_length[i]
    

    return stride_length,actual_length


def add_position(path: torch.Tensor, begin=0):
    path = path.float()  # 转换为浮点类型
    stride_length, actual_length = calculate_stride(path)
    begin = float(begin)  # 确保 begin 是浮点类型

    for i in range(path.shape[0]):
        for j in range(actual_length[i]):
            path[i][j] = path[i][j] + begin + j * stride_length[i]

    return path











if __name__ == "__main__":
    path = torch.tensor([[105311, 108573,     29,   1004,   3868,  20257,  14912,    307,   1071,
             15,  47443, 115277,  79409,  11759,    427,  20474, 210663,    257,
            716,    530,    427,  15736,   1002,  24975,    376, 129855,  51667,
             15,   2131,   1683,  10494,  87158, 136742,    999, 112213,   3868,
         113695,  58593,    530,  18210,  50497,    919,  28202,    336, 105311,
         242060,   7535,   4689,  47443,  65070,   9427,  75193,    530,  36790,
          32428,    368,  11468,    530,  10859,    461,  28202,     34,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3],
        [105311, 108573,     29,  47443, 115277,     15,  10393,   1002,  47443,
            760,  15449,     81,   4015,     15,  45691,  39343,    376,  28202,
            361,  34407,     15,    361,  65070,   9427,  23558,     10, 167785,
             17,  12941,  74177,   6242,    368,  92428,  31175,   1331, 125932,
            368,   3968,  28202,  26371,     15,   2131, 118536,  17625,   1485,
          17958,   1002,    267,   5579, 151908, 194236,  21438,    336, 105311,
         242060,   7535,   4689,  47443,  65070,   9427,  75193,    530,  36790,
          32428,    368,  11468,    530,  10859,    461,  28202,     34,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3],
        [105311, 108573,     29,  49262,   3868, 242527,    919,  28202,     15,
         115277,   1620,    329,  26733,   1485,    368,  16333,    361,  30222,
           1965,  25224,    361,  15278,    427,  14565,    368,  16333,   1485,
          12209,  20073,  57510,  32049,     17,  51786,   3868,  67791,     15,
          28202,  66893, 142657,  26942,   3269,    368, 168008,     15,  39222,
             15,    530,  99607,    336, 105311, 242060,   7535,   4689,  47443,
          65070,   9427,  75193,    530,  36790,  32428,    368,  11468,    530,
          10859,    461,  28202,     34,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3]])
    
    actual_length= calculate_stride(path)
    print(actual_length)

    new_path = add_position(path)
    print(new_path)