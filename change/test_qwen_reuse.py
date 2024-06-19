import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = "cuda" if torch.cuda.is_available() else "cpu"  # the device to load the model onto

checkpoint = "Qwen/Qwen1.5-4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left', padding=True, truncation=True)
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

print(text)

suffix = "Let's do it step by step"

model_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = [text],return_tensors="pt").to("cuda")
suffix_inputs = tokenizer._batch_encode_plus(batch_text_or_text_pairs = [suffix],return_tensors="pt").to("cuda")

# Ensure input IDs and attention masks are on the correct device
model_input_ids = model_inputs.input_ids.clone().detach().to(dtype=torch.long).to(device)
model_input_mask = model_inputs.attention_mask.clone().detach().to(dtype=torch.long).to(device)
suffix_input_ids = suffix_inputs.input_ids.clone().detach().to(dtype=torch.long).to(device)
suffix_input_mask = suffix_inputs.attention_mask.clone().detach().to(dtype=torch.long).to(device)
combine_ids = torch.cat((model_input_ids,suffix_input_ids),dim=1)
combine_mask = torch.cat((model_input_mask,suffix_input_mask),dim=1)
# Forward pass to get past key values
model_output = model.forward(input_ids=model_input_ids, use_cache=True, return_dict=True, attention_mask=model_input_mask)
past_key_values = model_output.past_key_values

# Generate new tokens using past key values
# Note: `max_length` should include the length of the original input plus the new tokens
outputs = model.generate(
    input_ids=combine_ids,
    attention_mask=combine_mask,
    past_key_values=past_key_values,
    use_cache=True,
    max_new_tokens=512,
)

# Decode the generated tokens
generated_tokens = outputs[0]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("Generated Text:", result)
