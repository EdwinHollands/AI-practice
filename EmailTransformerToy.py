# GET PYTORCH
import torch
import torch.nn as nn

# USE GPU IF POSS
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

# PARAMETERS ------------------------------------------
heads = 4 #number of attention heads in parallel
Q_dim = 8 #dimension of query space
emb_dim = heads*Q_dim #dimension of embedding space
block_size = 8   #i.e. 'context length', also called time T
batch_size = 32   #we will run multiple samples in parallel, B
learn_rate = 0.001
iters = 5000
interval = 500
prompt = 'Hello World'
gen_length = 50

# DATA --------------------------------------
with open('input.txt', 'r') as file:
    text = file.read()

# VOCABULARY -------------------------------------
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
# note will cause errors if fed characters that aren't in the text sample

# ENCODE/DECODE ------------------------------------------
stoi = {ch: i for i, ch in enumerate(vocab)} 
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# bijections between characters in the vocabulary and numbers, use dictionaries for O(1) time

# TENSOR and SPLIT -----------------------------------
data = torch.tensor(encode(text), dtype = torch.int64)
n=int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
# data is encoded as one long tensor and then first 90% allocated as training data

# SAMPLE SELECTION --------------------------------------------
def sample(dataset): #input 'train' for training
    data = train_data if dataset == 'train' else val_data
    samples=torch.randint(len(data)-block_size, (batch_size,)) # batch rand integers of max len-block
    inputs = torch.stack([data[i:i+block_size] for i in samples]) # batch x block
    targets = torch.stack([data[i+1:i+1+block_size] for i in samples]) # batch x block
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets # both are batch x block

# SELF ATTENTION HEAD ----------------------------------------
class Head(nn.Module):
    # INITIALISE ----
    def __init__(self, Q_dim):
        super().__init__()
        self.query = nn.Linear(emb_dim, Q_dim, bias=False) #questions
        self.key = nn.Linear(emb_dim, Q_dim, bias=False) #detecting relevance
        self.value = nn.Linear(emb_dim, Q_dim, bias=False) #answers
        # liner transforms from embedding space to query space
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        # lower triangular matrix of 1s used to mask future tokens from past

    def forward(self, inputs):
        B,T,C = inputs.shape
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        Aff = q @ k.transpose(-2,-1) * Q_dim **-0.5 # dot products -> batch x block x block, scaled by 1/root(query dim) to control variance
        Aff = Aff.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # sets upper triangle to -inf so tokens can't see the future
        Aff = nn.functional.softmax(Aff, dim=-1) # softmax: exp then normalise along rows
        out = Aff @ v # multiply affinities with values
        return out

# MULTI-HEAD ATTENTION -------------------------------------
class MultiHead(nn.Module):
    def __init__(self, heads, Q_dim):
        super().__init__()
        self.heads = nn.ModuleList([Head(Q_dim) for _ in range(heads)])
    
    def forward(self, inputs):
        return torch.cat([h(inputs) for h in self.heads], dim = -1)

# PEREPTRON / FEED FORWARD --------------------------------------
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim),nn.ReLU(),)
        #affine linear map followed by ReLU

    def forward(self, inputs):
        return self.net(inputs)

# BLOCK OF ATTENTION AND PERCEPTRON
class Block(nn.Module):
    def __init__(self, emb_dim, heads, Q_dim):  
        self.sa = MultiHead(heads, Q_dim)
        self.ffwd = FeedForward(emb_dim)

    def forward(self, inputs):
        inputs = self.sa(inputs)
        inputs = self.ffwd(inputs)
        return inputs

# DEFINE MODEL --------------------------------------------
class LanguageModel(nn.Module):
    # INITIALISE -----
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(block_size, emb_dim)
        #we associate each token and position to vectors in the embedding space
        self.lm_head = nn.Linear(emb_dim, vocab_size)
        # an affine linear map Ax+b that projects back from the embedding space to give logits for the vocabulary
        self.sa_head = MultiHead(heads, Q_dim)
        #a self attention head
        self.ffwd = FeedForward(emb_dim)

    # LOGITS AND LOSS -----
    def forward(self, inputs, targets=None):
        B , T = inputs.shape
        tok_emb = self.token_embedding_table(inputs)  # batch x block x dim
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # block x dim
        emb = tok_emb + pos_emb
        emb = self.sa_head(emb)
        emb = self.ffwd(emb)
        logits = self.lm_head(emb) # batch x block x vocab

        if targets is None:
            return logits, None # for generation

        B, T, C = logits.shape # B = batch, T is block or 'time', C is 'channels', in this case vocab
        logits = logits.view(B*T, C) # B*T = all tokens. Reshaping so cross entropy sees vocab as 2nd dim
        targets = targets.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets) # compare logits to targets
        return logits, loss #for training

    # GENERATION -------
    def generate(self, inputs, max):
        for _ in range(max):
            inputs_cropped = inputs[:, -block_size:]
            logits, loss = self(inputs_cropped)
            logits = logits[:,-1,:] # just take the final prediction since we aren't training
            probabilities = nn.functional.softmax(logits, dim=-1) # softmax to make a distribution
            next1 = torch.multinomial(probabilities, num_samples=1) # sample the distribution
            inputs = torch.cat((inputs, next1), dim=1) # append the new token
        return inputs

model = LanguageModel().to(device)

# TRAINING ----------------------------------------
optimiser = torch.optim.AdamW(model.parameters(), lr=learn_rate)

for i in range(iters):

    optimiser.zero_grad(set_to_none=True) # forget previous grads
    inputs, targets = sample('train') # pull a sample
    logits, loss = model(inputs, targets) # runs 'forward' pass
    loss.backward() # compute the gradient of the loss as a function of the parameters (back prop)
    if i % interval == 0:
        print(f"step {i} the loss is {loss.item()}")
    optimiser.step() # Update weights based on gradient flow scaled by learning rate


#TESTING ----------------------
inputs = torch.tensor([encode(prompt)], dtype = torch.int64, device=device)
new = decode(model.generate(inputs, gen_length)[0].tolist())
print(new)




