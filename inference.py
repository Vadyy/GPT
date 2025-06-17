import argparse
import torch
import tiktoken

from torch.nn import functional as F

from train_gpt2 import load_checkpoint, GPT

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024   # sequence length
    vocab_size: int = 50227  # number of tokens
    n_layer: int = 12        # number of layers
    n_head: int = 12         # number of heads
    n_embd: int = 768        # embedding dimension

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--finetune', action='store_true',
    #                     help='Run fine-tuning (post training) instead of pretraining.')
    parser.add_argument('--step', 
                        help='Model step')
    args = parser.parse_args()

    if not args.step:
        print("Use inference.py --step 5000")
        exit()

    model_name = "GPT"

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    print("Compiling model...")
    model = torch.compile(model)

    load_checkpoint(model, f"checkpoints/{model_name}-{args.step}.pth", device)

    while True:

        prompt = input("\n>>> ")

        enc = tiktoken.get_encoding('gpt2')
        x = enc.encode(prompt)
        x = torch.tensor(x).unsqueeze(0) 
        x = x.to(device)

        # T A L K

        print(enc.decode(x.squeeze().tolist()), end="")
        while x.size(1) < 250:
            with torch.no_grad():
                logits, loss = model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # only keeps the first 50 probabilities
                ix = torch.multinomial(topk_probs, 1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

                print(enc.decode([xcol[0, 0].item()]).replace("��", "'"), end="", flush=True)