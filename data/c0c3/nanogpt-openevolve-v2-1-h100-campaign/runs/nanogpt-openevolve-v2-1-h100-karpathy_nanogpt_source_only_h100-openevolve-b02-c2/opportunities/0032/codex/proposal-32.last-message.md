MECHANISM: Compute-matched early-context reallocation

HYPOTHESIS: Using 152-token windows in the first three local layers and 120-token windows in the final three will lower val_bpb below 0.983275 by extending the observed advantage of allocating more local context to earlier layers while preserving the six-layer average of 136 tokens.

INTENDED_EDIT: Replace uniform 192-token local attention with 152-token early and 120-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: The compute-matched 144-early/128-late design achieved the best result, 0.983275, outperforming uniform 136 at 0.983317 and the reversed 128-early/144-late allocation at 0.983641; increasing the same early-layer bias tests whether that directional improvement continues.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = 3 * long_window // 32
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
        early_short_window = 19 * long_window // 256
        late_short_window = 15 * long_window // 256
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
WINDOW_PATTERN = "SSSL" # 192-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 152-token early and 120-token late local layers; full context at layers 4 and 8
>>>>>>> REPLACE