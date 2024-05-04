# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# checkpoint = "/home/lipz/BloomzLink/bloomz3b/bloomz-3b"

# # 加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


# print("Tokenizer Configuration:")
# print(tokenizer.name_or_path)  # 显示分词器的路径或名称
# print(tokenizer.model_max_length)  # 显示模型允许的最大输入长度

# print("\nModel Configuration:")
# print(model.config)  # 打印模型的全部配置信息

# # 准备输入数据
# inputs = tokenizer.encode("Instruction: Write a high-quality answer for the given question using only the following relevant search results.", return_tensors="pt").to("cuda")
# print(inputs)
# print("----------------\n")
# # 生成输出

#     # def generate(
#     #     self,
#     #     inputs: Optional[torch.Tensor] = None,
#     #     generation_config: Optional[GenerationConfig] = None,
#     #     logits_processor: Optional[LogitsProcessorList] = None,
#     #     stopping_criteria: Optional[StoppingCriteriaList] = None,
#     #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#     #     synced_gpus: Optional[bool] = None,
#     #     assistant_model: Optional["PreTrainedModel"] = None,
#     #     streamer: Optional["BaseStreamer"] = None,
#     #     negative_prompt_ids: Optional[torch.Tensor] = None,
#     #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#     #     **kwargs,
#     # ) -> Union[GenerateOutput, torch.LongTensor]:
# outputs = model.generate(inputs,num_beams=5,early_stopping=True)
# print(outputs)

# # 解码并打印输出结果
# print("\nGenerated Text:")
# print(tokenizer.decode(outputs[0]))


# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # 示例输入
# instruction_ids = tokenizer.encode("The science of today is the technology of tomorrow.", return_tensors="pt")

# # 假设有预先计算的 past_key_values（通常在连续生成中使用）
# # 假设 instruction_kv 已经定义
# instruction_kv = None  # 仅为示例，实际应为之前生成过程中得到的past_key_values

# # 生成文本
# output = model.generate(
#     inputs=instruction_ids,
#     past_key_values=instruction_kv,
#     use_cache=True,
#     max_new_tokens=128,
#     min_length=128  # 确保生成至少有128个token
# )

# # 解码生成的文本
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigscience/bloomz-7b1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("###Instruction: Write a high-quality answer for the given question using only the following relevant search results.\n",
        "###Chunk 1:In his early twenties, Steve Jobs visited India to seek enlightenment and to experiment with psychedelic drugs, which he later claimed profoundly influenced his creative strategies and business practices at Apple.\n###Question:How did Steve Jobs' experiences and decisions shape the development and success of Apple?\n", return_tensors="pt").to("cuda")
outputs = model.generate(inputs,max_new_tokens = 128)
print(tokenizer.decode(136742))
