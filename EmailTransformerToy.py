import torch
import torch.nn as nn
#First we download our dataset

#now retrieve it to process
#with open('emails.txt', 'r') as file:
    #data = file.read()

#for now, let's keep it simple
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam hendrerit nisi sed sollicitudin pellentesque. Nunc posuere purus rhoncus pulvinar aliquam. Ut aliquet tristique nisl vitae volutpat. Nulla aliquet porttitor venenatis. Donec a dui et dui fringilla consectetur id nec massa. Aliquam erat volutpat. Sed ut dui ut lacus dictum fermentum vel tincidunt neque. Sed sed lacinia lectus. Duis sit amet sodales felis. Duis nunc eros, mattis at dui ac, convallis semper risus. In adipiscing ultrices tellus, in suscipit massa vehicula eu."

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

#lets encode our data as a tensor
data = torch.tensor(encode(text), dtype = torch.int64)
#and split it into training and validation sets
n=int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#now lets pick a block size i.e. 'context length'
block_size = 3

#we will run multiple samples in parallel
batch_size = 2

#now make a process to pull out a chunk of the data. Input string 'train' for training or 'val' for validation.
def sample(set):
    data = train_data if set == 'train' else val_data
    samples=torch.randint(len(data)-block_size, (batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in samples])
    targets = torch.stack([data[i+1:i+1+block_size] for i in samples])
    # we create 2d tensors [batches*[samples]]
    return inputs, targets

#so we feed samples into the transformer

print(sample('train'))




