MECHANISM: Hardware-aligned depth-tapered local context

HYPOTHESIS: A monotone 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983275 by approximating the moderate early-context bias while keeping every window 8-token aligned.

INTENDED_EDIT: Replace uniform 136-token local attention with a compute-matched depth taper, preserving full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: The aligned 144-early/128-late design achieved the best 0.983275, while the stronger 152/120 bias and non-8-aligned 140/132 refinement regressed; a three-level aligned taper tests a gentler transition without changing aggregate attention compute.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = 17 * long_window // 256
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
WINDOW_PATTERN = "SSSL" # 136-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 144/144/136/136/128/128 local taper; full context at layers 4 and 8
>>>>>>> REPLACE