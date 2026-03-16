import torch
import torch.nn as nn

device        = 'cuda' if torch.cuda.is_available() else 'cpu' # use gpu if poss
#PARAMETERS
block_size = 32   #i.e. 'context length', also called time steps?
batch_size = 32   #we will run multiple samples in parallel
learn_rate = 0.001
iters = 4000
interval = 300
prompt = 'hello world'
gen_length = 50

#First we get our dataset
with open('input.txt', 'r') as file:
    text = file.read()

#Find our vocabulary in terms of characters
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

#now we need to build an encoder and decoder

#encode = lambda s: [vocab.index(c) for c in s] <- O(n), not good

#O(1) version using dictionaries
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

#lets encode our data as a tensor
data = torch.tensor(encode(text), dtype = torch.int64)
#and split it into training and validation sets
n=int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#now make a process to pull out a chunk of the data. Input string 'train' for training or 'val' for validation.
def sample(dataset):
    data = train_data if dataset == 'train' else val_data
    samples=torch.randint(len(data)-block_size, (batch_size,))
    inputs = torch.stack([data[i:i+block_size] for i in samples])
    targets = torch.stack([data[i+1:i+1+block_size] for i in samples])
    inputs = inputs.to(device)
    targets = targets.to(device)
    # we create 2d tensors [batches*[samples]]
    return inputs, targets

#so we feed samples into the transformer
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__() # 'square' since number of embeddings = embedding dim = vocab size so each token is embedded as a vocab-size dimensional vector
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx/inputs and targets both are 2d tensor batch*time
        logits = self.token_embedding_table(idx)  # batch*time*vocab_size

        if targets is None:
            return logits, None #for generation

        B, T, C = logits.shape # B is batch size, T is sequence length or time, C is vocab size
        logits = logits.view(B*T, C) # B*T = number of tokens, vocab size. Reshaping so cross entropy sees vocab as 2nd dim
        targets = targets.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss #for training

        #we need a way to generate
    def generate(self, idx, max):
        for _ in range(max):
            logits, loss = self(idx)
            logits = logits[:,-1,:] # just take the final prediction since we aren't training
            probabilities = nn.functional.softmax(logits, dim=-1) #softmax to make a distribution
            next1 = torch.multinomial(probabilities, num_samples=1) # sample the distribution
            idx = torch.cat((idx, next1), dim=1) # append the new token
        return idx

model = BigramLanguageModel(vocab_size).to(device)

#Now we need to train
optimiser = torch.optim.AdamW(model.parameters(), lr=learn_rate)

for i in range(iters):

    optimiser.zero_grad(set_to_none=True)
    inputs, targets = sample('train')
    logits, loss = model(inputs, targets)
    loss.backward()
    if i % interval == 0:
        print(f"step {i} the loss is {loss.item()}")
    optimiser.step()


inputs = torch.tensor([encode(prompt)], dtype = torch.int64, device=device)
new = decode(model.generate(inputs, gen_length)[0].tolist())
print(new)




