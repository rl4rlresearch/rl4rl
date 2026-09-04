MECHANISM: Native full-causal FA3 fast path for global layers

HYPOTHESIS: Using FA3’s native `(-1, -1)` full-causal mode only for the two existing global layers will preserve the best SSSL architecture while increasing token throughput enough to beat `val_bpb` 0.987174.

INTENDED_EDIT: Keep six half-context layers and two global layers, but represent global attention with FA3’s native full-causal window and preserve that representation for the forced-final global layer.

EVIDENCE: The all-global native-causal run reached 44.47% MFU and 482.3M tokens, showing that `(-1, -1)` uses an efficient kernel path; this isolates that path improvement without discarding the SSSL pattern that achieved the best 0.987174 result.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (-1, -1), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = char_to_window["L"]
>>>>>>> REPLACE