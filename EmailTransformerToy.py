import torch
import torch.nn as nn
#First we download our dataset

#now retrieve it to process
with open('input.txt', 'r') as file:
    text = file.read()

#for now, let's keep it simple
#text = "Hello world. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam hendrerit nisi sed sollicitudin pellentesque. Nunc posuere purus rhoncus pulvinar aliquam. Ut aliquet tristique nisl vitae volutpat. Nulla aliquet porttitor venenatis. Donec a dui et dui fringilla consectetur id nec massa. Aliquam erat volutpat. Sed ut dui ut lacus dictum fermentum vel tincidunt neque. Sed sed lacinia lectus. Duis sit amet sodales felis. Duis nunc eros, mattis at dui ac, convallis semper risus. In adipiscing ultrices tellus, in suscipit massa vehicula eu."

#let's check a sample
#print(f"First hundred characters are: {text[:100]}")

#Find our vocabulary in terms of characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

#print(f"Vocabulary size: {vocab_size}, list of characters: {vocab}")

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

#now lets pick a block size i.e. 'context length', also called time steps?
block_size = 8

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__() # number of embeddings = embedding dim = vocab size so each token is embedded as a vocab-size dimensional vector
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx/inputs and targets both are 2d tensor batch*time
        logits = self.token_embedding_table(idx)  # batch*time*vocab_size

        if targets is None:
            return logits, None

        B, T, C = logits.shape # B is batch size, T is sequence length or time, C is vocab size
        logits = logits.view(B*T, C) # B*T = number of tokens, vocab size. Reshaping so cross entropy sees vocab as 2nd dim
        targets = targets.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

        #we need a way to generate
    def generate(self, idx, max):
        for _ in range(max):
            logits, loss = self(idx)
            logits = logits[:,-1,:] # just take the final prediction since we aren't training
            probabilities = nn.functional.softmax(logits, dim=-1) #softmax to make a distribution
            next = torch.multinomial(probabilities, num_samples=1) # sample the distribution
            idx = torch.cat((idx, next), dim=1) # append the new token
        return idx

model = BigramLanguageModel(vocab_size)

#Now we need to train
optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)

batch_size = 4
for _ in range(1000):
    optimiser.zero_grad(set_to_none=True)
    inputs, targets = sample('train')
    logits, loss = model(inputs, targets)
    loss.backward()
    optimiser.step()
print(loss.item())


inputs = torch.tensor([encode('hello world')], dtype = torch.int64)
new = decode(model.generate(inputs, 50)[0].tolist())
print(new)




