"""
SpectralLM: Unified spectral language model.

Architecture:
    Input tokens
        ↓
    SpectralEmbedding       Laplacian eigenvector init + MoPE sinusoidal PE
                            Random sign flip per token during training
        ↓
    [Loop R times]:
        SpectralTransformerLayer   shared weights across loops
            EGA energy gate on values (scalar per position, 1 linear layer)
            Causal attention + ALiBi-style per-head locality bias
            FF network
                ↓
        EntityMemory               Titans-style fast-weight associative memory
            Surprise = prediction error (||M·k - v||) as gradient-magnitude proxy
            Update: momentum + weight-decay, gated by surprise
            Output: additive to hidden state
        ↓
    LayerNorm
        ↓
    LM head:        partially tied to amplitude embedding
                    logits = h_real @ amplitude.weight.T + lm_head_extra(h_imag)
    Waveform head:  separate small projection → next-token amplitude

Training modes:
    'waveform': waveform completion pre-training
    'lm':       language modeling
    'joint':    both losses simultaneously
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import SpectralEmbedding


class SpectralTransformerLayer(nn.Module):
    """
    Shared transformer layer with EGA and ALiBi-style per-head locality bias.

    EGA (Energy Gate on value Aggregation):
        gate = sigmoid(W_gate · x_norm)   shape (B, T, 1)
        v_gated = v * gate                applied before attention aggregation
    Suppresses low-confidence value contributions at low-energy positions.

    Locality bias (ALiBi-style):
        score[b, h, i, j] -= |pos_i - pos_j| / σ_h
    σ_h is a per-head learnable scale initialized from the embedding's log_sigma
    band structure (structural heads → large σ → near-global; semantic → small σ → local).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        log_sigma_init: torch.Tensor,   # (embed_dim,) from SpectralEmbedding.log_sigma
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = math.sqrt(self.head_dim)

        self.in_proj     = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj    = nn.Linear(dim, dim, bias=True)
        self.energy_gate = nn.Linear(dim, 1, bias=True)    # EGA: scalar gate per position
        self.attn_drop   = nn.Dropout(dropout)

        # Per-head locality scale — initialized from embedding band structure.
        # half = embed_dim // n_heads: head h covers real dims [h*half:(h+1)*half].
        half = self.head_dim // 2
        with torch.no_grad():
            sigma_init     = torch.exp(log_sigma_init.detach())   # (embed_dim,)
            per_head_sigma = torch.stack([
                sigma_init[h * half : (h + 1) * half].mean()
                for h in range(n_heads)
            ])
        self.log_sigma_head = nn.Parameter(torch.log(per_head_sigma))  # (n_heads,)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.energy_gate.weight)
        nn.init.zeros_(self.energy_gate.bias)

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        H, HD   = self.n_heads, self.head_dim

        residual = x
        x_norm   = self.norm1(x)

        # QKV projections → split heads
        qkv = self.in_proj(x_norm)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, HD).transpose(1, 2)            # (B, H, T, HD)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)

        # EGA: energy gate on values before aggregation
        gate = torch.sigmoid(self.energy_gate(x_norm))      # (B, T, 1)
        v    = v * gate.unsqueeze(1)                        # (B, H, T, 1) → (B, H, T, HD)

        # Scaled dot-product + ALiBi-style locality bias
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # (B, H, T, T)

        pos   = torch.arange(T, device=x.device).float()
        dist  = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()           # (T, T)
        sigma = torch.exp(self.log_sigma_head)                         # (H,)
        bias  = -(dist.unsqueeze(0) / (sigma.view(H, 1, 1) + 1e-6))  # (H, T, T)
        scores = scores + bias.unsqueeze(0)

        if causal_mask is not None:
            scores = scores + causal_mask.view(1, 1, T, T)

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                    # (B, H, T, HD)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x   = residual + self.out_proj(out)

        x = x + self.ff(self.norm2(x))
        return x


class EntityMemory(nn.Module):
    """
    Titans-style associative fast-weight memory with chunked causal scan.

    Maintains a (B, mem_dim, mem_dim) fast-weight matrix M per sequence.
    The sequence is processed in chunks of size `chunk_size` (default 16).
    Within each chunk all positions share the same M (from before the chunk),
    then M is updated once in bulk at the chunk boundary. This gives strict
    causality at chunk boundaries and reduces Python loop overhead ~chunk_size×
    vs a token-by-token scan (128→8 iterations for seq_len=128, chunk=16).

    Per chunk c covering positions [s, s+C):
        K_c = normalize(key_proj(h_c))       (B, C, Md)
        V_c = val_proj(h_c)                  (B, C, Md)
        M_c = M · K_c^T                      (B, Md, C)  — readout (same M for all C)
        E_c = V_c − M_c^T                    (B, C, Md)  — prediction error
        gate = sigmoid(||E_c|| + gate_bias)  (B, C, 1)
        M   = γ · M + (gate · E_c)^T · K_c  (B, Md, Md) — bulk outer-product update

    L_memory = mean(||E_c||²) across all positions.
    M is detached before each chunk read/write; gradient flows only through
    key_proj, val_proj, out_proj, gate_bias for the current chunk's computation.
    """

    def __init__(
        self,
        dim:          int,
        mem_dim:      int,
        momentum:     float = 0.9,
        weight_decay: float = 0.01,
        chunk_size:   int   = 16,
    ):
        super().__init__()
        self.mem_dim      = mem_dim
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.chunk_size   = chunk_size

        self.key_proj  = nn.Linear(dim, mem_dim, bias=False)
        self.val_proj  = nn.Linear(dim, mem_dim, bias=False)
        self.out_proj  = nn.Linear(mem_dim, dim, bias=True)
        self.gate_bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.val_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        M: 'torch.Tensor | None' = None,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]':
        """
        h:  (B, T, dim)           hidden states from current loop iteration
        M:  (B, mem_dim, mem_dim) fast-weight state (None → initialize zeros)

        Returns: (output, mem_loss, M_new)
            output:   (B, T, dim)           memory readout, added to hidden
            mem_loss: scalar                mean prediction error (L_memory term)
            M_new:    (B, mem_dim, mem_dim) updated fast-weight state
        """
        B, T, _ = h.shape
        Md, C   = self.mem_dim, self.chunk_size
        gamma   = self.momentum - self.weight_decay

        if M is None:
            M = h.new_zeros(B, Md, Md)

        K = F.normalize(self.key_proj(h), dim=-1)   # (B, T, Md)
        V = self.val_proj(h)                          # (B, T, Md)

        readout_chunks = []
        sq_error_chunks = []

        for s in range(0, T, C):
            e = min(s + C, T)
            K_c = K[:, s:e]   # (B, C, Md)
            V_c = V[:, s:e]   # (B, C, Md)

            # Causal readout — all positions in chunk use M from before this chunk
            M_c = torch.bmm(M.detach(), K_c.transpose(1, 2))   # (B, Md, C)
            m_c = M_c.transpose(1, 2)                            # (B, C, Md)

            E_c  = V_c - m_c                                     # (B, C, Md)
            surp = E_c.norm(dim=-1, keepdim=True)                # (B, C, 1)
            gate = torch.sigmoid(surp + self.gate_bias)          # (B, C, 1)

            # Bulk outer-product update for the chunk
            weighted = gate * E_c                                 # (B, C, Md)
            M_delta  = torch.bmm(weighted.transpose(1, 2), K_c)  # (B, Md, Md)
            M        = gamma * M.detach() + M_delta

            readout_chunks.append(self.out_proj(m_c))            # (B, C, dim)
            sq_error_chunks.append(E_c.pow(2).mean())

        output   = torch.cat(readout_chunks, dim=1)              # (B, T, dim)
        mem_loss = torch.stack(sq_error_chunks).mean()
        return output, mem_loss, M


class SpectralLM(nn.Module):
    """
    Unified spectral language model with shared-weight transformer loops
    and Titans-style entity memory.

    Args:
        vocab:             list of token strings
        embed_dim:         per-modality dim (output hidden = 2*embed_dim)
        n_layers:          legacy alias for n_loops
        n_heads:           attention heads (must divide 2*embed_dim)
        max_seq_len:       maximum sequence length
        dropout:           dropout rate
        band_init:         banded frequency initialization
        acoustic_init:     acoustic fallback amplitude init
        laplacian_eigvecs: (V, V) Laplacian eigenvectors for amplitude init
        n_loops:           R — how many times to apply the shared transformer layer
        mem_dim:           fast-weight memory dimension (0 to disable EntityMemory)
    """

    def __init__(
        self,
        vocab:             list,
        embed_dim:         int = 64,
        n_layers:          int = 3,
        n_heads:           int = 8,
        max_seq_len:       int = 512,
        dropout:           float = 0.1,
        band_init:         bool = True,
        acoustic_init:     bool = True,
        laplacian_eigvecs: 'torch.Tensor | None' = None,
        n_loops:           'int | None' = None,
        mem_dim:           int = 32,
    ):
        super().__init__()

        self.vocab_size = len(vocab)
        self.embed_dim  = embed_dim
        self.dim        = 2 * embed_dim
        self.n_loops    = n_loops if n_loops is not None else n_layers
        self.mem_dim    = mem_dim

        assert self.dim % n_heads == 0, \
            f"2*embed_dim ({self.dim}) must be divisible by n_heads ({n_heads})"

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding = SpectralEmbedding(
            vocab_size        = self.vocab_size,
            embed_dim         = embed_dim,
            max_seq_len       = max_seq_len,
            vocab             = vocab,
            band_init         = band_init,
            acoustic_init     = acoustic_init,
            laplacian_eigvecs = laplacian_eigvecs,
        )

        # ── Single shared transformer layer (looped n_loops times) ───────────
        log_sigma_init = self.embedding.log_sigma.data.clone()
        self.shared_layer = SpectralTransformerLayer(
            dim            = self.dim,
            n_heads        = n_heads,
            log_sigma_init = log_sigma_init,
            dropout        = dropout,
        )

        # ── Entity memory (optional, disabled when mem_dim=0) ────────────────
        self.entity_memory = EntityMemory(self.dim, mem_dim) if mem_dim > 0 else None

        self.norm = nn.LayerNorm(self.dim)

        # ── Output heads ─────────────────────────────────────────────────────
        # Waveform head: separate small projection (D, not 2D target)
        self.waveform_head = nn.Linear(self.dim, embed_dim)

        # LM head: partially tied to amplitude embedding.
        # The real half of hidden projects through amplitude (same basis as input),
        # the imaginary half through an independent learned projection.
        self.lm_head_extra = nn.Linear(embed_dim, self.vocab_size, bias=True)

        self._init_output_heads()

    def _init_output_heads(self):
        nn.init.normal_(self.waveform_head.weight, std=0.02)
        nn.init.zeros_(self.waveform_head.bias)
        nn.init.normal_(self.lm_head_extra.weight, std=0.02)
        nn.init.zeros_(self.lm_head_extra.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(
        self,
        tokens:    torch.Tensor,   # (B, T) long
        positions: torch.Tensor,   # (B, T) long
        mode:      str = 'lm',
    ) -> dict:
        """
        Returns dict with:
            'hidden':    (B, T, dim)      final hidden states
            'lm_logits': (B, T, vocab)    next-token logits  (if mode includes lm)
            'waveform':  (B, T, D)        predicted next amplitude (if waveform mode)
            'embedding': (B, T, 2D)       input waveform (for gradient consistency loss)
            'mem_loss':  scalar           mean memory prediction error across loops
        """
        B, T = tokens.shape

        x             = self.embedding(tokens, positions)
        embedding_out = x

        causal_mask    = self._causal_mask(T, tokens.device)
        M              = None
        total_mem_loss = torch.tensor(0.0, device=tokens.device)

        for _ in range(self.n_loops):
            x = self.shared_layer(x, causal_mask)
            if self.entity_memory is not None:
                mem_out, mem_loss, M = self.entity_memory(x, M)
                x              = x + mem_out
                total_mem_loss = total_mem_loss + mem_loss

        x = self.norm(x)

        out = {
            'hidden':    x,
            'embedding': embedding_out,
            'mem_loss':  total_mem_loss / max(self.n_loops, 1),
        }

        if mode in ('lm', 'joint'):
            h_real = x[:, :, :self.embed_dim]                    # (B, T, D)
            h_imag = x[:, :, self.embed_dim:]                    # (B, T, D)
            out['lm_logits'] = (
                h_real @ self.embedding.amplitude.weight.T        # tied part
                + self.lm_head_extra(h_imag)                      # free part
            )

        if mode in ('waveform', 'joint'):
            out['waveform'] = self.waveform_head(x)              # (B, T, D)

        return out

    def param_count(self) -> dict:
        total       = sum(p.numel() for p in self.parameters())
        embed       = sum(p.numel() for p in self.embedding.parameters())
        transformer = sum(p.numel() for p in self.shared_layer.parameters())
        memory      = (sum(p.numel() for p in self.entity_memory.parameters())
                       if self.entity_memory is not None else 0)
        heads       = (sum(p.numel() for p in self.waveform_head.parameters()) +
                       sum(p.numel() for p in self.lm_head_extra.parameters()))
        return {
            'total': total, 'embedding': embed,
            'transformer': transformer, 'memory': memory, 'heads': heads,
        }
