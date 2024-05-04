import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,LogitsProcessorList,MinLengthLogitsProcessor,StoppingCriteriaList,MaxLengthCriteria
from accelerate import Accelerator

#I think path is the instruction+chunk+query  but we do not change the instruction position encoding
# path[path_num , path_id]


# def calculate_harmonic_mean(path:torch.Tensor,):
#     sum = 0.0
#     path_num,_ = path.shape
#     for i in range(path_num):
#         length = len(path[i])
#         sum = sum+1/length

#     mean =  path_num/sum
#     print("sum:",sum)

#     return mean

# 直白的来说，这种位置方式把左填充的那种会生成大量<pad>平均分布到整个序列，从而可以形象的比作弹簧。这种方法减少了填充token对生成质量的影响
def prepare_position(path:torch.Tensor):
    path_num,max_length= path.shape
    mean = 0.0
    sum = 0.0
    # strart = 0
    # if(instruction_length>0):
    #     strart = instruction_length+1
    # else:
    #     print("please remember input instruction length")
    # position_tensor = torch.ones(path_num, max_length,dtype=torch.int64)
    #识别 <pad>和其他token 然后分配位置
    chunk_pad  = torch.zeros(path_num,dtype=torch.int64)
    for i in range(path_num):
        pad_num = 0
        #默认之前的tokenizer是左填充
        for j in range(max_length):
            if path[i][j] == 3:
                pad_num = pad_num+1
            else:
                break
        chunk_pad[i] = pad_num
    for i in range(path_num):
        actual_length = max_length - chunk_pad[i]
        sum = sum+1/actual_length

    mean = path_num/sum
    print(mean,chunk_pad,max_length)


    new_path = torch.zeros(path_num,max_length,dtype=torch.int64)
    for i in range(path_num):

        step = int(mean/(max_length-chunk_pad[i])+1)+1
        print(f"loop{i},step:{step}")
        j = 0
        for j in range(max_length-chunk_pad[i]):
            #j将要剩下的位置装不下还剩下的词
            if((max_length-j*step)<(max_length-chunk_pad[i]-j)):
                break
            new_path[i][j*step] = path[i][j+chunk_pad[i]]
        print("###j:",j)
        for k in range(max_length-chunk_pad[i]-j+1):
            new_path[i][(j-1)*step+k] = path[i][chunk_pad[i]+j+k-1]



                


    for i in range(path_num):
        for j in range(max_length):
            if(new_path[i][j] == 0):
                new_path[i][j] = 3

          

    return new_path

#函数问题：弹簧不是很平均

if __name__ == "__main__":

    path = torch.tensor([[     3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3, 105311, 108573,     29,   1004,   3868,  20257,
          14912,    307,   1071,     15,  47443, 115277,  79409,  11759,    427,
          20474, 210663,    257,    716,    530,    427,  15736,   1002,  24975,
            376, 129855,  51667,     15,   2131,   1683,  10494,  87158, 136742,
            999, 112213,   3868, 113695,  58593,    530,  18210,  50497,    919,
          28202,    336, 105311, 242060,   7535,   4689,  47443,  65070,   9427,
          75193,    530,  36790,  32428,    368,  11468,    530,  10859,    461,
          28202,     34],
        [     3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3, 105311, 108573,     29,  47443, 115277,     15,
          10393,   1002,  47443,    760,  15449,     81,   4015,     15,  45691,
          39343,    376,  28202,    361,  34407,     15,    361,  65070,   9427,
          23558,     10, 167785,     17,  12941,  74177,   6242,    368,  92428,
          31175,   1331, 125932,    368,   3968,  28202,  26371,     15,   2131,
         118536,  17625,   1485,  17958,   1002,    267,   5579, 151908, 194236,
          21438,    336, 105311, 242060,   7535,   4689,  47443,  65070,   9427,
          75193,    530,  36790,  32428,    368,  11468,    530,  10859,    461,
          28202,     34],
        [     3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3,      3,      3,
              3,      3,      3,      3,      3,      3,      3, 105311, 108573,
             29,  49262,   3868, 242527,    919,  28202,     15, 115277,   1620,
            329,  26733,   1485,    368,  16333,    361,  30222,   1965,  25224,
            361,  15278,    427,  14565,    368,  16333,   1485,  12209,  20073,
          57510,  32049,     17,  51786,   3868,  67791,     15,  28202,  66893,
         142657,  26942,   3269,    368, 168008,     15,  39222,     15,    530,
          99607,    336, 105311, 242060,   7535,   4689,  47443,  65070,   9427,
          75193,    530,  36790,  32428,    368,  11468,    530,  10859,    461,
          28202,     34]])

    position_tensor = prepare_position(path)
    print(position_tensor)

