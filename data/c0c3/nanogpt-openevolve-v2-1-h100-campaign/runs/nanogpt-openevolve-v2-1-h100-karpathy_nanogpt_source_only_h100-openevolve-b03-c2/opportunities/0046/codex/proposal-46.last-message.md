MECHANISM: Front-loaded local-context hybrid

HYPOTHESIS: Using 512-token local attention before the intermediate full-context layer and 384-token attention afterward will reduce `val_bpb` below 0.983618.

INTENDED_EDIT: Keep full context at layers 4 and 8, widen the first three local layers to 512 tokens, and retain the best uniform design’s 384-token window for the final three local layers.

EVIDENCE: The 512/256 front-loaded split reached 0.983620, essentially matching uniform 384, while the reverse 256/512 split regressed to 0.984213. This suggests early context is more valuable, while uniform 512’s 0.984156 argues against widening the late layers too.

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
        early_short_window = long_window // 4
        late_short_window = 3 * long_window // 16
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
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 512 to 384
>>>>>>> REPLACE