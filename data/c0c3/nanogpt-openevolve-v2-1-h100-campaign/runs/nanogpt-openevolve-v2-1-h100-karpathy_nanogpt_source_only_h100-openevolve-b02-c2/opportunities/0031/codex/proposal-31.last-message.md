MECHANISM: Reverse depth-progressive local-context allocation

HYPOTHESIS: Using 144-token windows in the first three local layers and 128-token windows in the final three will lower val_bpb below 0.983317 by favoring early context formation while preserving the aggregate attention compute of uniform 136-token windows.

INTENDED_EDIT: Replace uniform 512-token local attention with 144-token early and 128-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: Uniform 136-token windows achieved 0.983317, while the compute-matched 128-early/144-late allocation regressed to 0.983641; reversing that allocation directly tests whether local context is more valuable in earlier layers.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
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
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 512-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 144-token early and 128-token late local layers; full context at layers 4 and 8
>>>>>>> REPLACE