import os
import math
import time
import inspect
import tiktoken
import argparse
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from dataclasses import dataclass

# ----------------------------------

# https://www.youtube.com/watch?v=l8pRSuU81PU 
# 2:47:00 / 4:01:25

# -------
# Ideas -
# -------
# Make a model design it's own tokenizer then train a better one with it.
# -------
# After teaching a model Reasoning, make it generate a hyper stream of conciousness that fit
# as the "thoughts" of the model's Reasoning output. Basically train it to have a conciousness
# stream of a conciousness stream of an output text (thoughts of thoughts of text), i.e: 2 linear reasonings.
# Special sauce: this second reasoning should be incomprehensible to us, but extremely optimized for "it"
# -------
# The previous idea might need an implementation of Reinforcement Learning. Try coming up with an algorithm ourselves.
# -------
# Custom Tokenizer
# -------

# !-Warning-! ver. Alpha Early Access

@dataclass
class GPTConfig:
    block_size: int = 1024   # sequence length
    vocab_size: int = 50227  # number of tokens
    n_layer: int = 12        # number of layers
    n_head: int = 12         # number of heads
    n_embd: int = 768        # embedding dimension

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) # mask
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # original attention (non flash)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh') # remove the approximation later on (why is it here? we're copying gpt2)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),

        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

#
##
####
#####
class DataLoaderStream:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding('gpt2')
        
        self.ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        self.ds_iter = iter(self.ds)
        
        # This buffer will accumulate tokens from streamed samples
        self.token_buffer = []
        print("Streaming dataset loaded.")

    def next_batch(self):
        # We need B*T+1 tokens so that we can create input (x) and target (y) pairs
        required_tokens = self.B * self.T + 1
        
        # Fill the buffer until we have enough tokens
        while len(self.token_buffer) < required_tokens:
            try:
                sample = next(self.ds_iter)
            except StopIteration:
                # If we run out of data, reinitialize the iterator
                self.ds_iter = iter(self.ds)
                sample = next(self.ds_iter)
                
            # Tokenize the new text and extend the token buffer
            tokens = self.enc.encode(sample["text"])
            self.token_buffer.extend(tokens)

        # Slice off the required tokens for one batch
        buf_tokens = self.token_buffer[:required_tokens]
        # Remove the used tokens from the buffer
        self.token_buffer = self.token_buffer[required_tokens:]
        
        # Convert the tokens to a tensor
        buf = torch.tensor(buf_tokens)
        # Prepare input and target: x is tokens[:-1] and y is tokens[1:]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        return x, y

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('vady.txt', 'r', encoding="utf-8") as f:
            text = f.read()
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f'Model saved to {filepath}')

def load_model(model, filepath, device):
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    print(f'Model loaded from {filepath} and moved to {device}')

def save_checkpoint(model, optimizer, step, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f'\nCheckpoint saved to {filepath}')

def load_checkpoint(model, filepath, device, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f'Checkpoint loaded from {filepath}, last trained step {step}')
    return step

###
#####
#######
###############################################
#######
#####
###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune', action='store_true',
                        help='Run fine-tuning (post training) instead of pretraining.')
    parser.add_argument('--resume', 
                        help='Continue pre-training [step]')
    parser.add_argument('--name', 
                        help='Name the model')
    args = parser.parse_args()

    # ---

    model_name = "GPT"

    if args.name:
        model_name = args.name

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    total_batch_size = 524288 # 2**19, ~0.5M

    B = 8
    T = 1024

    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    print("Loading dataset...")
    train_loader = DataLoaderStream(B=B, T=T)
    torch.set_float32_matmul_precision('high')

    ##

    ##

    # My  B R A I N

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    print("Compiling model...")
    model = torch.compile(model)

    ##

    ##

    #

    max_lr = 6e-4
    min_lr = max_lr * 0.1 # 10%
    warmup_steps = 10
    max_steps = 10000

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

    step = 0
    if args.resume is not None and args.resume.isnumeric():
        step = args.resume
        step = load_checkpoint(model, f"checkpoints/{model_name}-{step}.pth", device, optimizer)
        step = step+1
        do_generation_at_start = step

        print(f'Resuming training from step {step}')

    #

    # L E A R N I N G

    # Tur ning on m y con ciou sness

    for step in range(step, max_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient norm clipping (prevents model shock)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # for timing loss_accum
        t1 = time.perf_counter()
        dt = t1 - t0
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        if step % 10 == 0:
            print(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

        if (step % 100 == 0 or do_generation_at_start) and step != 0:
            do_generation_at_start = None
            enc = tiktoken.get_encoding('gpt2')
            x = enc.encode(" ")
            x = torch.tensor(x).unsqueeze(0) 
            x = x.to(device)

            max_length = random.randint(64, 1024)

            print(enc.decode([x.item()]), end="")
            while x.size(1) < max_length:
                with torch.no_grad():
                    logits, loss = model(x)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # only keeps the first 50 probabilities
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    x = torch.cat((x, xcol), dim=1)

                    print(enc.decode([xcol[0, 0].item()]), end="", flush=True)
            print("\n")
        if step % 100 == 0 and step != 0:
            save_checkpoint(model, optimizer, step, f"checkpoints/{model_name}-{step}.pth")

    save_model(model, f"models/{model_name}-{step}.pth")


    # import sys; sys.exit(0)

    enc = tiktoken.get_encoding('gpt2')
    x = enc.encode("I am")
    x = torch.tensor(x).unsqueeze(0) 
    x = x.to(device)

    max_length = 5000

    # T A L K

    print(enc.decode(x.squeeze().tolist()), end="")
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # only keeps the first 50 probabilities
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

            print(enc.decode([xcol[0, 0].item()]), end="", flush=True)



    # tokens = x[0, :max_length].tolist()
    # decoded = enc.decode(tokens)
    # print("> ", decoded)
