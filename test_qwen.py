import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ [ "CUDA_VISIBLE_DEVICES" ] = "0,1,2,3"
device = "cuda" if torch.cuda.is_available() else "cpu" # the device to load the model onto

checkpoint = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,padding_side = 'left',padding=True,truncation=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt",padding = "max_length",max_length=50).to(device)
print(f"model_inputs{model_inputs}")
outputs = model.generate(model_inputs["input_ids"], max_new_tokens=512)
generated_tokens = outputs[0] # Get only the generated tokens
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Generated Text:", result)

