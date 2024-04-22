# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "/home/lipz/BloomzLink/bloomz3b/bloomz-3b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))

