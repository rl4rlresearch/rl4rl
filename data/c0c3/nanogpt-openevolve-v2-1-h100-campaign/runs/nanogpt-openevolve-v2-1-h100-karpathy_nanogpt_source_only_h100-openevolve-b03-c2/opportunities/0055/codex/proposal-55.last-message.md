MECHANISM: Upper-side early-context interpolation with late-layer pruning

HYPOTHESIS: Using 480-token pre-global windows and 256-token post-global windows will reduce `val_bpb` below 0.983549.

INTENDED_EDIT: Widen the first three local-attention layers from 384 to 480 tokens, narrow the final three local layers to 256 tokens, and retain full attention at layers 4 and 8.

EVIDENCE: The verified 464/256 design achieved the best `val_bpb` of 0.983549, slightly beating 448/256 at 0.983556, while 512/256 regressed to 0.983620; the prior 480 attempt produced no verifiable result, so this unresolved point remains the most informative upper-side interpolation.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = 3 * long_window // 16
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = 15 * long_window // 64
        late_short_window = long_window // 8
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
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=3/16 context
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 480 to 256
>>>>>>> REPLACE