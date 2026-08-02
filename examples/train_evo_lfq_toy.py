#!/usr/bin/env uv run
# /// script
# dependencies = [
#   "torch",
#   "tqdm",
#   "fire",
#   "einops",
# ]
# ///

import time
from pathlib import Path
import sys
from tqdm.auto import trange

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vector_quantize_pytorch import EvoLFQ

def divisible_by(num, den):
    return (num % den) == 0

def decode_logits_to_str(logits):
    indices = logits.argmax(dim = -1)
    return ''.join([chr(int(c.item())) for c in indices])

def main(
    goal = 'attention is all you need',
    train_steps = 2000,
    generations = 300,
    pop_size = 256,
    mutation_rate = 0.02,
    tournament_size = 3,
    elitism_count = 2,
    lr = 3e-3,
    seed = 42
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running EvoLFQ String Autoencoder Toy Task on {device} ===")

    num_chars = len(goal)
    target_ascii = torch.tensor([ord(c) for c in goal], dtype = torch.long, device = device)

    # structured phrase generator (expanded 45-word text manifold)

    words = [
        'attention', 'is', 'all', 'you', 'need', 'deep', 'learning', 'neural',
        'network', 'model', 'latent', 'vector', 'space', 'quantum', 'transformer',
        'diffusion', 'generative', 'evolutionary', 'algorithm', 'quantization',
        'codebook', 'encoder', 'decoder', 'optimization', 'gradient', 'backprop',
        'recurrent', 'convolution', 'sequence', 'embedding', 'representation',
        'autoencoder', 'intelligence', 'artificial', 'supervision', 'contrastive',
        'entropy', 'manifold', 'stochastic', 'hyperparameter', 'probability', 'tensor',
        'residual', 'token', 'alignment'
    ]

    def generate_phrase_batch(batch_size = 64):
        phrases = []
        for _ in range(batch_size):
            phrase = ""
            while len(phrase) < num_chars:
                w = words[torch.randint(0, len(words), (1,)).item()]
                phrase += (w + " ")
            phrase = phrase[:num_chars].ljust(num_chars)
            phrases.append([ord(c) for c in phrase])
        return torch.tensor(phrases, dtype = torch.long, device = device)

    num_compressed_codebooks = 12
    bits_per_codebook = 8
    compressed_latent_dim = num_compressed_codebooks * bits_per_codebook

    class StringEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, 32)
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_chars * 32, 256),
                nn.GELU(),
                nn.Linear(256, compressed_latent_dim)
            )

        def forward(self, x):
            return self.net(self.emb(x))

    class StringDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(compressed_latent_dim, 256),
                nn.GELU(),
                nn.Linear(256, num_chars * 256)
            )

        def forward(self, codes):
            flat_codes = codes.reshape(codes.shape[0], -1)
            return self.net(flat_codes).reshape(-1, num_chars, 256)

    evo_lfq = EvoLFQ(
        encoder = StringEncoder(),
        decoder = StringDecoder(),
        codebook_size = 256,
        num_codebooks = num_compressed_codebooks,
        pop_size = pop_size,
        mutation_rate = mutation_rate,
        tournament_size = tournament_size,
        elitism_count = elitism_count,
        generations = generations
    ).to(device)

    # 1. Train autoencoder on structured text manifold

    opt = AdamW(evo_lfq.parameters(), lr = lr)
    evo_lfq.train()

    print(f"\n1. Training String Autoencoder + LFQ for {train_steps} steps...")
    pbar = trange(train_steps)
    for _ in pbar:
        opt.zero_grad()
        batch_text = generate_phrase_batch(64)
        logits, _, aux_loss = evo_lfq(batch_text)
        rec_loss = F.cross_entropy(logits.view(-1, 256), batch_text.view(-1))
        loss = rec_loss + aux_loss
        loss.backward()
        opt.step()

        pbar.set_description(f"Rec Loss: {rec_loss.item():.4f} | Aux Loss: {aux_loss.item():.4f}")

    print(f"Autoencoder training completed. Final Rec Loss: {rec_loss.item():.4f}\n")

    # 2. Evolutionary Latent Search in LFQ binary space

    print(f"2. Evolving latent binary search for target phrase '{goal}'...")
    evo_lfq.eval()

    def fitness_fn(decoded_logits, compressed_bits):
        log_probs = F.log_softmax(decoded_logits, dim = -1)
        target_log_probs = log_probs[:, torch.arange(num_chars, device = device), target_ascii]
        return target_log_probs.sum(dim = -1)

    start_time = time.time()
    for gen_idx, result in enumerate(evo_lfq.evolve(
        fitness_fn = fitness_fn,
        shape = (num_compressed_codebooks, bits_per_codebook),
        return_best_decoded = True
    )):
        should_log = gen_idx == 0 or divisible_by(gen_idx + 1, 50) or (gen_idx + 1) == generations
        if should_log:
            best_str = decode_logits_to_str(result.best_decoded)
            print(f"Gen {gen_idx + 1:3d}/{generations} | Max Fitness: {result.best_fitness:.4f} | Evolved: '{best_str}'")

    elapsed = time.time() - start_time
    print(f"\nEvolution completed in {elapsed:.2f}s!")

    final_str = decode_logits_to_str(result.best_decoded)
    print(f"Final Evolved String: '{final_str}'")

if __name__ == "__main__":
    fire.Fire(main)
