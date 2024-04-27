# SuperPE_prompt

this is a simple replication of a thesis-based algorithm
Base paper link https://arxiv.org/abs/2404.06910

In this paper, the author do his work base on bloomz mmodel, which code is in transformer package.  transformers/models/bloom/modeling_bloom.py   https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py

model param  https://huggingface.co/bigscience

checkpoint = "/home/lipz/BloomzLink/bloomz3b/bloomz-3b"    this is my own weight path, you should change it to your path when you run it GPUS

if you want to use Accelerate package to accelerate your code and make your code run on multi GPU, use accelerate lanuch path_to.py  tutorial https://zhuanlan.zhihu.com/p/684526775


所有 generation()可以传入的其他参数所在路径图片
![alt text](/pic/generation_config.png)
![alt text](/pic/model_info.png)
![alt text](/pic/model_info2.png)

![alt text](/pic/attention_matrix.png)
![alt text](/pic/image.png)
