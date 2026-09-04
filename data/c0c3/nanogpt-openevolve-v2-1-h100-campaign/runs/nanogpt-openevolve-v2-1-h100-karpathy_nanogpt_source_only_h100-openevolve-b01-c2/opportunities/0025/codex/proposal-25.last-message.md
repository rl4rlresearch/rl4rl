MECHANISM: Compute-neutral progressive local-attention funnel

HYPOTHESIS: Redistributing the seven local windows from uniform 256-token attention to progressively expanding 128–512-token windows, while preserving the same 1,792-token aggregate local-attention budget and final global anchor, will retain roughly 525M training tokens and lower val_bpb below 0.992854.

INTENDED_EDIT: Replace seven uniform 256-token windows with 128, 160, 192, 224, 256, 320, and 512-token windows before the unchanged full-context final layer.

EVIDENCE: The 256-token single-anchor design is best at 0.992854, while uniform 128-token attention loses quality and a 512-token pre-anchor bridge remains viable; reallocating rather than increasing attention compute tests whether narrow early layers and broader late layers combine their benefits.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 8
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
        local_windows = (128, 160, 192, 224, 256, 320, 512)
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                local_window = local_windows[min(layer_idx, len(local_windows) - 1)]
                window_sizes.append((local_window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # progressive 128-512-token local layers, then one full-context anchor
>>>>>>> REPLACE