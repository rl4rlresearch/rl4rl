MECHANISM: Refined compute-neutral staged short-context attention

HYPOTHESIS: A 832/1024/1216-token progression will lower val_bpb below 0.991835 while retaining at least 475M training tokens.

INTENDED_EDIT: Redistribute each three-layer short-attention budget from uniform 1024-token windows to a milder staged progression without changing total attention-window FLOPs.

EVIDENCE: The 768/1024/1280 schedule achieved the best val_bpb, 0.991835, while uniform windows reached 0.992110 and the steeper 512/1024/1536 schedule regressed to 0.992198; refining the successful slope tests the apparent optimum between uniform and overly steep staging.

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
        long_window = config.sequence_len
        assert pattern == "SSSL"
        # Preserve the total short-attention budget while gradually expanding
        # context toward the representation handed to each full-context layer.
        short_windows = (
            13 * long_window // 32,
            long_window // 2,
            19 * long_window // 32,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE