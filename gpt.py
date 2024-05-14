import pathlib # for dealing with file paths
import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
block_size = 8 # set the block size ( maximum context length for predictions)
batch_size = 32 # set the batch size ( how many blocks to train on at once)
max_iters = 3000 # set the number of iterations to train for
eval_interval = 300 # set the number of iterations to evaluate the model
learnign_rate = 1e-3 # set the learning rate
eval_iters = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if a GPU is available
n_embed = 32 # set the embedding size

print(device)

path = pathlib.Path().resolve() # get the path to the current directory
#read in all the words
with open(path.joinpath('input.txt'), 'r', encoding='utf-8') as f:
    text = f.read() # read the file

chars = sorted(list(set(text))) # get all the unique characters in the text
vocab_size = len(chars) # get the number of unique characters

stoi = {ch: i for i, ch in enumerate(chars)} # create a dictionary mapping characters to integers
itos = {i: ch for i, ch in enumerate(chars)} # create a dictionary mapping integers to characters
encode = lambda x: [stoi[ch] for ch in x] # create a function to encode a string into integers
decode = lambda x: ''.join([itos[i] for i in x]) # create a function to decode a list of integers into a string

data = torch.tensor(encode(text), dtype=torch.long) # encode the text into integers and convert it to a tensor

n = int(0.9*len(data)) # get the number of training samples, 90% of the data
train_data = data[:n] # get the training samples
val_data = data[n:] # get the validation samples

def get_batch(split): 
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get random starting indices for the batch
    x = torch.stack([data[i:i+block_size] for i in ix]) # create the input tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # create the target tensor
    x, y = x.to(device), y.to(device) # move the data to the proper device
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (batch_size, block_size, n_embed) (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (block_size, n_embed) (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size) (B, T, C)
        if(targets is None):
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # get the predictions
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # get the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model = model.to(device) # move the model to the proper device

optimizer = torch.optim.Adam(model.parameters(), lr=learnign_rate) # create an optimizer object

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}, Train loss: {losses["train"]:.4f}, Val loss: {losses["val"]:.4f}')

    xb, yb = get_batch('train')

    # evaluate the loss
    logits,loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens= 500)[0].tolist()))


