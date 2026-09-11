MECHANISM: Native full-attention kernel path

HYPOTHESIS: Representing the two full-context layers with FlashAttention’s native unrestricted window will preserve the winning SSSL receptive field while improving throughput enough to reduce val_bpb below 0.993365.

INTENDED_EDIT: Keep 512-token local windows, but encode full-context attention as `(-1, -1)` instead of the equivalent `(2048, 0)` sliding window.

EVIDENCE: The best design uses full attention at layers 3 and 7; removing one gained only 4.7M tokens and worsened val_bpb to 0.994122, motivating a numerical optimization of full attention rather than reducing global mixing.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        short_window = config.sequence_len // 4
        full_window = (-1, -1)
        char_to_window = {"L": full_window, "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = full_window
>>>>>>> REPLACE