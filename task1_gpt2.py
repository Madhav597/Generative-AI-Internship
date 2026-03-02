from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "Artificial Intelligence will"

inputs = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    no_repeat_ngram_size=2
)

text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(text)