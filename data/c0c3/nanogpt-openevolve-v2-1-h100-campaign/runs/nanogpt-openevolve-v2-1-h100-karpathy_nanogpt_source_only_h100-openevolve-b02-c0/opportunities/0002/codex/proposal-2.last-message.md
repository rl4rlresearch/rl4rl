MECHANISM: True full-causal attention fast path

HYPOTHESIS: Replacing all sliding-window layers with FA3’s native full-causal mode will beat val_bpb 0.995558 by avoiding the inefficient local-attention path while giving every layer complete context.

INTENDED_EDIT: Use `(-1, -1)` for global attention and make every layer global.

EVIDENCE: Reducing six local windows from 1024 to 512 tokens unexpectedly cut throughput from 497.0M to 404.2M tokens and worsened val_bpb to 1.009175, indicating that smaller sliding windows are counterproductive on this kernel.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
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
        full_window = (-1, -1)
        short_window = (config.sequence_len // 2, 0)
        char_to_window = {"L": full_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = full_window
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "L"    # all layers use FA3's native full-causal fast path
>>>>>>> REPLACE