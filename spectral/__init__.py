"""
Spectral Language Model

A unified waveform-based model where:
- Tokens are initialized from character-level spectral composition
- Positional encoding uses MoPE (learned frequency + locality per dimension)
- Frequency bands are partitioned hierarchically (structural / expression / semantic)
- Phase encodes position within each structural scale
- One unified model, no separate Quinn + Transformer components

Architecture overview:
    SpectralEmbedding:     character-composed init + MoPE basis
    SpectralEncoder:       multi-head attention operating in frequency space
    SpectralDecoder:       waveform completion head
"""
