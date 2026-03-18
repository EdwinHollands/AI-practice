# GET PYTORCH
import torch
import torch.nn as nn
import time
import os

# USE GPU IF POSS
device        = 'cuda' if torch.cuda.is_available() else 'cpu'

# activate the venv first
# MIGHT NEED TO COPY TO TERMINA: C:\Users\echol\Documents\Coding\AI-practice\gpt_env\Scripts\activate


# PARAMETERS ------------------------------------------
heads = 6 #number of attention heads in parallel per block
layers = 6 #number of attention/perceptron blocks
ff_scalar = 4
Q_dim = 64 #dimension of query space
emb_dim = Q_dim*heads #dimension of embedding space
context_block = 256   #i.e. 'context length', also called time T
batch_size = 64   #we will run multiple samples in parallel, B
learn_rate = 0.0001
dropout = 0.5 # blocks some weights in training to prevent overfitting
iters = 10000
interval = 500
gen_length = 10000
patience = 5

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
    samples=torch.randint(len(data)-context_block, (batch_size,)) # batch rand integers of max len-block
    inputs = torch.stack([data[i:i+context_block] for i in samples]) # batch x block
    targets = torch.stack([data[i+1:i+1+context_block] for i in samples]) # batch x block
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
        self.register_buffer('tril', torch.tril(torch.ones(context_block,context_block)))
        # lower triangular matrix of 1s used to mask future tokens from past
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        B,T,C = inputs.shape
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        Aff = q @ k.transpose(-2,-1) * Q_dim **-0.5 # dot products -> batch x block x block, scaled by 1/root(query dim) to control variance
        Aff = Aff.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # sets upper triangle to -inf so tokens can't see the future
        Aff = nn.functional.softmax(Aff, dim=-1) # softmax: exp then normalise along rows
        Aff = self.dropout(Aff)
        out = Aff @ v # multiply affinities with values
        return out

# MULTI-HEAD ATTENTION -------------------------------------
class MultiHead(nn.Module):
    def __init__(self, heads, Q_dim):
        super().__init__()
        self.heads = nn.ModuleList([Head(Q_dim) for _ in range(heads)])
        self.project = nn.Linear(heads * Q_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        out = torch.cat([h(inputs) for h in self.heads], dim = -1)
        return self.dropout(self.project(out))

# PEREPTRON / FEED FORWARD --------------------------------------
class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_scalar):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_scalar*emb_dim),
            nn.ReLU(),
            nn.Linear(ff_scalar*emb_dim, emb_dim),
            nn.Dropout(dropout))
        #affine linear map followed by ReLU

    def forward(self, inputs):
        return self.net(inputs)

# LAYER OF ATTENTION AND PERCEPTRON
class Layer(nn.Module):
    def __init__(self, emb_dim, heads, Q_dim):
        super().__init__()
        self.sa = MultiHead(heads, Q_dim)
        self.ffwd = FeedForward(emb_dim, ff_scalar)
        self.ln1  = nn.LayerNorm(emb_dim) #layer norms fix means and deviations for each token then does a linear transform
        self.ln2  = nn.LayerNorm(emb_dim)

    def forward(self, inputs):
        inputs = inputs + self.sa(self.ln1(inputs)) #residuals are important for deep networks!
        inputs = inputs + self.ffwd(self.ln2(inputs))
        return inputs

# DEFINE MODEL --------------------------------------------
class LanguageModel(nn.Module):
    # INITIALISE -----
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(context_block, emb_dim)
        # an affine linear map Ax+b that projects back from the embedding space to give logits for the vocabulary
        self.layers = nn.Sequential(*[Layer(emb_dim, heads, Q_dim) for _ in range(layers)])
        self.ln = nn.LayerNorm(emb_dim)
        #we associate each token and position to vectors in the embedding space
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    # LOGITS AND LOSS -----
    def forward(self, inputs, targets=None):
        B , T = inputs.shape
        tok_emb = self.token_embedding_table(inputs)  # batch x block x dim
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # block x dim
        emb = tok_emb + pos_emb
        emb = self.layers(emb)
        emb = self.ln(emb)
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
            inputs_cropped = inputs[:, -context_block:]
            logits, loss = self(inputs_cropped)
            logits = logits[:,-1,:] # just take the final prediction since we aren't training
            probabilities = nn.functional.softmax(logits, dim=-1) # softmax to make a distribution
            next1 = torch.multinomial(probabilities, num_samples=1) # sample the distribution
            inputs = torch.cat((inputs, next1), dim=1) # append the new token
        return inputs

model = LanguageModel().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# LOSS ESTIMATION --------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            inputs, targets = sample(split)
            logits, loss = model(inputs, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# MODE SELECTION ----------------------------
print("What would you like to do?")
print("  1. Train a new model")
print("  2. Generate from saved model")
mode = input("Enter 1 or 2: ").strip()

# TRAINING MODE -----------------------------
if mode == '1':
    # CHOOSE OPTIMISER -------------------------
    optimiser = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=iters)
    # WARMUP ESTIMATE ----------------------------------------
    warmup_steps = 10
    print(f"Running {warmup_steps} warmup steps to estimate training time...")
    warmup_start = time.time()
    for i in range(warmup_steps):
        optimiser.zero_grad(set_to_none=True)
        inputs, targets = sample('train')
        logits, loss = model(inputs, targets)
        loss.backward()
        optimiser.step()
    warmup_elapsed = time.time() - warmup_start

    time_per_step = warmup_elapsed / warmup_steps
    estimated_total = time_per_step * iters
    est_mins = int(estimated_total // 60)
    est_secs = int(estimated_total % 60)

    print(f"Estimated training time: {est_mins}m {est_secs}s for {iters} iterations")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        exit()

    # TRAINING ----------------------------------------
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0 
    for i in range(iters+1):

        optimiser.zero_grad(set_to_none=True) # forget previous grads
        inputs, targets = sample('train') # pull a sample
        logits, loss = model(inputs, targets) # runs 'forward' pass
        loss.backward() # compute the gradient of the loss as a function of the parameters (back prop)
        if i % interval == 0 and i>0:
            losses = estimate_loss()
            elapsed = time.time() - start_time
            steps_done = i
            steps_remaining = iters - i
            time_per_step = elapsed / steps_done
            eta_seconds = time_per_step * steps_remaining
            eta_mins = int(eta_seconds // 60)
            eta_secs = int(eta_seconds % 60)
            print(f"step {i}/{iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | ETA {eta_mins}m {eta_secs}s")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                torch.save(model.state_dict(), 'model.pt')
                print(f"  ↳ new best val loss {best_val_loss:.4f}, model saved")
            else:
                patience_counter += 1
                print(f"  ↳ no improvement, patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at step {i}")
                    break
        optimiser.step() # Update weights based on gradient flow scaled by learning rate
        scheduler.step() # update the learning rate
    total_time = time.time() - start_time
    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)
    print(f"Training complete in {total_mins}m {total_secs}s")

#GENERATION MODE ----------------------
elif mode == '2':
    if not os.path.exists('model.pt'):
        print("No saved model.")
        exit()
    model.load_state_dict(torch.load('model.pt', map_location=device))
    print("Loaded model from model.pt")
    model.eval()
    prompt = input("Enter a prompt (or press Enter for zero): ").strip()
    if prompt == '':
        inputs = torch.zeros((1, 1), dtype=torch.int64, device=device)
    else:
        prompt = ''.join([c for c in prompt if c in stoi])
        inputs = torch.tensor([encode(prompt)], dtype = torch.int64, device=device)
    with torch.no_grad():
        generated = decode(model.generate(inputs, gen_length)[0].tolist())
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated)

    print(f"Generated {gen_length} tokens to output.txt")
    os.startfile(output_path)
else:
    print("Invalid option.")
    exit()