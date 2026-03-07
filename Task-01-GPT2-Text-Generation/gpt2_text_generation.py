from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Prompt
prompt = "Artificial Intelligence will"

inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:\n")
print(generated_text)
