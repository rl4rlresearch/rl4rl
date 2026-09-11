MECHANISM: Scale-selective hybrid positional attention

HYPOTHESIS: Leaving half of each full-context layer’s query/key channels unrotated, while restoring the proven 704/1024/1344 short-window progression, will lower val_bpb below 0.991729 while retaining at least 475M training tokens.

INTENDED_EDIT: Use full RoPE in short-context layers but a mixed positional/content subspace in full-context layers, allowing their learned projections to perform position-independent semantic retrieval without weakening local order modeling.

EVIDENCE: The 704/1024/1344 schedule achieved the best result, 0.991729, while further window refinements regressed. The low-rank prefix-memory path reached only 0.994246 and reduced throughput, motivating global-context representation inside the existing full-attention computation instead of another compressed context path.

<<<<<<< SEARCH
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
=======
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def apply_partial_rotary_emb(x, cos, sin):
    """Rotate half the head channels while retaining a position-free subspace."""
    assert x.ndim == 4 and x.shape[3] % 4 == 0
    d = x.shape[3] // 4
    x1, x2, x_content = x[..., :d], x[..., d:2 * d], x[..., 2 * d:]
    # Subsample the original frequency bank so the rotated half still spans
    # both short- and long-wavelength positional frequencies.
    partial_cos, partial_sin = cos[..., ::2], sin[..., ::2]
    y1 = x1 * partial_cos + x2 * partial_sin
    y2 = x1 * (-partial_sin) + x2 * partial_cos
    return torch.cat([y1, y2, x_content], 3)
>>>>>>> REPLACE

<<<<<<< SEARCH
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
=======
        self.is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = self.is_long_layer or precedes_long_layer
>>>>>>> REPLACE

<<<<<<< SEARCH
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
=======
        cos, sin = cos_sin
        if self.is_long_layer:
            q = apply_partial_rotary_emb(q, cos, sin)
            k = apply_partial_rotary_emb(k, cos, sin)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
>>>>>>> REPLACE

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
=======
        long_window = config.sequence_len
        assert pattern == "SSSL"
        # Retain the best verified compute-neutral progression, concentrating
        # context toward each full-attention handoff.
        short_windows = (
            11 * long_window // 32,
            long_window // 2,
            21 * long_window // 32,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
>>>>>>> REPLACE