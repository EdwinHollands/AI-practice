import torch
#First we download our dataset

#now retrieve it to process
#with open('emails.txt', 'r') as file:
    #data = file.read()

text = "hello world"

#let's check a sample
print(f"First hundred characters are: {text[:100]}")

#Find our vocabulary in terms of characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}, list of characters: {vocab}")

#now we need to build an encoder and decoder
encode = lambda s: [vocab.index(c) for c in s]
decoded_chars = lambda n: [vocab[k] for k in n]
decode = lambda n: "".join(decoded_chars(n))

print(encode("hello world"))
print(decode(encode("hello world")))

#lets encode our data as a tensor
data = torch.tensor(encode(text), dtype = torch.long)