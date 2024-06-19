from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import torch

# Define CUDA devices and memory
cuda_list = [0, 1, 2, 3]  # Using integers for GPU identifiers
memory = '8.0GiB'
model_path = '/home/lipz/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8'
no_split_module_classes = LlamaForCausalLM._no_split_modules

# Correct max_memory dictionary with device identifiers
max_memory = {cuda: memory for cuda in cuda_list}

# Load model configuration
config = LlamaConfig.from_pretrained(model_path)

# Initialize model with empty weights
with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)

# Tie model weights
model.tie_weights()

# Infer device map
device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

# Load checkpoint in model with the correct device map
load_checkpoint_in_model(model, model_path, device_map=device_map, offload_folder='/home/lipz/vllm_cache', offload_state_dict=True)

# Dispatch model to devices
model = dispatch_model(model, device_map=device_map, offload_dir='/home/lipz/vllm_cache')

# Load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Prepare input data
sents = ['你是谁']
ids = tokenizer(sents, max_length=1800, padding=True, truncation=True, return_tensors="pt")

# Ensure input tensors are on the correct device
for key, value in ids.items():
    ids[key] = value.to('cuda:0')  # Assuming you're using cuda:0 for input tensors

# Disable gradient calculations
torch.set_grad_enabled(False)
model.eval()

# Generate outputs
outputs = model.generate(**ids, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
