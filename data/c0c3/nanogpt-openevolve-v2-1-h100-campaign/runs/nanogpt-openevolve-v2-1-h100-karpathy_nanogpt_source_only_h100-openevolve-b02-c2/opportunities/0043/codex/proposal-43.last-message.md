MECHANISM: Full-context value-embedding alignment

HYPOTHESIS: Relocating the layer-4 value embedding to layer 3 while using the best verified layer-3/layer-8 full-context layout will lower val_bpb below 0.982662 by injecting token-specific values during the first global mixing operation without changing parameter count.

INTENDED_EDIT: Restore the verified 144/144/136/136/128/128 local taper with full context at layers 3 and 8, and move one value embedding and its gate from layer 4 to layer 3.

EVIDENCE: Moving first full context from layer 4 to layer 3 improved val_bpb from 0.983152 to 0.982662, despite leaving the alternating value embedding at layer 4; this motivates aligning that fixed-cost resource with the newly superior global layer.

<<<<<<< SEARCH
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
=======
def has_ve(layer_idx, n_layer):
    """Use four value embeddings, aligning one with the early full-context layer."""
    if n_layer == 8:
        return layer_idx in (1, 2, 5, 7)
    return layer_idx % 2 == (n_layer - 1) % 2
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = 9 * long_window // 128
        late_short_window = long_window // 16
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                short_window = early_short_window if layer_idx < config.n_layer // 2 else late_short_window
                window_sizes.append((short_window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            17 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
        window_sizes = []
        short_idx = 0
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                window_sizes.append((short_windows[short_idx], 0))
                short_idx += 1
        assert short_idx == len(short_windows)
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144-token early and 128-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/136/136/128/128 local taper; full context at layers 3 and 8
>>>>>>> REPLACE