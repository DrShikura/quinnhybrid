"""
Text generation for SpectralLM.

At inference time, the waveform head is unused.
Generation is purely: embedding → transformer → LM head → sample.

Supports:
    greedy:      always pick argmax (deterministic, often repetitive)
    temperature: sample with temperature scaling (default, best quality)
    top_k:       restrict sampling to top-k logits
    top_p:       nucleus sampling — restrict to smallest set summing to p
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def _top_k_top_p_filter(logits: torch.Tensor,
                         top_k: int = 0,
                         top_p: float = 1.0) -> torch.Tensor:
    """Filter logits using top-k and/or nucleus (top-p) sampling."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens where cumulative prob exceeds top_p
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))

        # Scatter back to original ordering
        logits = torch.zeros_like(logits).scatter(
            -1, sorted_idx, sorted_logits
        )

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt:      str  = "",
    max_new:     int  = 128,
    temperature: float = 0.8,
    top_k:       int  = 50,
    top_p:       float = 0.95,
    stop_on_eos: bool = True,
    device:      str  = 'cpu',
) -> str:
    """
    Generate tokens autoregressively from a prompt.

    Args:
        model:       SpectralLM instance (eval mode)
        tokenizer:   PythonStructuralTokenizer
        prompt:      source code prefix to continue
        max_new:     maximum new tokens to generate
        temperature: sampling temperature (lower = more focused)
        top_k:       top-k filtering (0 = disabled)
        top_p:       nucleus sampling threshold
        stop_on_eos: stop when EOS token is generated
        device:      torch device string

    Returns:
        Generated structural token sequence as a readable string
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    # Encode prompt
    if prompt.strip():
        token_ids = tokenizer.encode(prompt, add_special=True)
    else:
        token_ids = [tokenizer.bos_id]

    generated = list(token_ids)

    for _ in range(max_new):
        # Truncate to model's max_seq_len
        ctx = generated[-model.embedding.max_seq_len:]

        tokens    = torch.tensor([ctx], dtype=torch.long, device=dev)
        positions = torch.arange(len(ctx), dtype=torch.long, device=dev).unsqueeze(0)

        # Forward — LM mode only at inference
        out    = model(tokens, positions, mode='lm')
        logits = out['lm_logits'][0, -1, :]     # (vocab_size,) last position

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
            logits = _top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = logits.argmax().item()

        generated.append(next_id)

        if stop_on_eos and next_id == tokenizer.eos_id:
            break

    # Decode: convert token IDs back to structural token names
    token_names = tokenizer.decode(generated)
    return ' '.join(token_names)


@torch.no_grad()
def generate_beam(
    model,
    tokenizer,
    prompt:      str = "",
    max_new:     int = 64,
    n_beams:     int = 4,
    device:      str = 'cpu',
) -> str:
    """
    Beam search generation — more coherent but slower.
    Better for structured outputs where greedy makes locally wrong choices.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    if prompt.strip():
        token_ids = tokenizer.encode(prompt, add_special=True)
    else:
        token_ids = [tokenizer.bos_id]

    # Each beam: (score, token_list)
    beams = [(0.0, list(token_ids))]

    for _ in range(max_new):
        candidates = []

        for score, seq in beams:
            if seq[-1] == tokenizer.eos_id:
                candidates.append((score, seq))
                continue

            ctx       = seq[-model.embedding.max_seq_len:]
            tokens    = torch.tensor([ctx], dtype=torch.long, device=dev)
            positions = torch.arange(len(ctx), dtype=torch.long, device=dev).unsqueeze(0)

            out    = model(tokens, positions, mode='lm')
            logits = out['lm_logits'][0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            # Expand top-n_beams
            top_lp, top_ids = log_probs.topk(n_beams)

            for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                candidates.append((score + lp, seq + [tid]))

        # Keep top beams
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:n_beams]

        # Early stop if all beams ended
        if all(seq[-1] == tokenizer.eos_id for _, seq in beams):
            break

    best_seq = beams[0][1]
    return ' '.join(tokenizer.decode(best_seq))
