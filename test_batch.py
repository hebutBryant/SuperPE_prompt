import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
torch.set_printoptions(threshold=1000000)

# 初始化Accelerator
accelerator = Accelerator()

checkpoint = "bigscience/bloomz-3b"
# device = "cuda:0" if torch.cuda.is_available() else "cpu" 
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='right')
model = AutoModelForCausalLM.from_pretrained(checkpoint)
# model.to_bettertransformer()
# model = model.to("cuda:0")

prompts = [
    "Translate to English: Je t’aime.",
    "Describe the plot of The Friends."
]

suffix = "Let's do it step by step"

# 设置最大长度，确保为8的倍数，使用左侧填充
inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=10).to(accelerator.device)
# Assuming `inputs` is a dictionary containing tensors:
input_ids = inputs['input_ids']  # This gets the actual input tensor
suffix_inputs = tokenizer(suffix, return_tensors="pt").to(accelerator.device)
suffix_input_ids = suffix_inputs['input_ids']
batch_size = input_ids.shape[0]
suffix_input_ids_expanded = suffix_input_ids.expand(batch_size, -1)
combined_input_ids = torch.cat([input_ids, suffix_input_ids_expanded], dim=1)
print(f"combined_input_ids:{combined_input_ids}")

# Now, prepare it with the accelerator:
model, input_ids = accelerator.prepare(model, combined_input_ids)

# Use the tensor directly:
outputs = model.generate(combined_input_ids, max_new_tokens=256)

# Decode and print outputs:
print("Generate Text:", tokenizer.decode(outputs[0], skip_special_tokens=True))


