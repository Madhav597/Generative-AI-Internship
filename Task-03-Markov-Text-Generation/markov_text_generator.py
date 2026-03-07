import random

# Sample training text
text = """
Artificial intelligence is transforming the world.
Artificial intelligence helps machines learn from data.
Machine learning is a part of artificial intelligence.
AI will shape the future of technology.
"""

# Split text into words
words = text.split()

# Create Markov Chain dictionary
markov_chain = {}

for i in range(len(words) - 1):
    word = words[i]
    next_word = words[i + 1]

    if word not in markov_chain:
        markov_chain[word] = []

    markov_chain[word].append(next_word)

# Function to generate text
def generate_text(length=20):
    word = random.choice(words)
    result = [word]

    for _ in range(length - 1):
        if word in markov_chain:
            word = random.choice(markov_chain[word])
            result.append(word)
        else:
            word = random.choice(words)
            result.append(word)

    return " ".join(result)

# Generate text
generated = generate_text(30)

print("Generated Text:\n")
print(generated)
