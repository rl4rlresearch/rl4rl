MECHANISM: Local-to-global multiscale attention funnel

HYPOTHESIS: Six 512-token layers followed by a 1024-token penultimate layer and one full-context final layer will retain more than 503.3M training tokens while improving val_bpb below 0.993870.

INTENDED_EDIT: Replace the current two-anchor layout with six quarter-context layers, one half-context bridge layer, and one final full-context anchor.

EVIDENCE: Reference Design 3’s single final full-context anchor achieved the best val_bpb, 0.993870, while removing that anchor worsened val_bpb to 1.001821; adding a cheaper half-context bridge tests richer pre-anchor mixing without restoring the costly intermediate full-context layer.

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
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 2
        short_window = long_window // 4
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # proven layout: two full-context anchor layers
=======
WINDOW_PATTERN = "SSSSSSML" # six quarter-context layers, half-context bridge, full anchor
>>>>>>> REPLACE