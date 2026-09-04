MECHANISM: Fine-grained early-context interpolation with late-layer pruning

HYPOTHESIS: Using 464-token windows in the three local layers before layer-4 global attention and 256-token windows afterward will reduce `val_bpb` below 0.983556.

INTENDED_EDIT: Restore full attention at layers 4 and 8, set early local windows to 464 tokens, retain 256-token late windows, and preserve all optimizer settings.

EVIDENCE: The 448/256 design achieved 0.983556, outperforming both 384/256 at 0.983738 and 512/256 at 0.983620; 464 tokens refines the bracket near its empirically favored interior.

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
        early_short_window = 29 * long_window // 128
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
WINDOW_PATTERN = "SSLSSSSL" # 384-token local layers; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 464 to 256
>>>>>>> REPLACE