MECHANISM: Native dense dispatch for existing global-attention layers

HYPOTHESIS: Routing only the two existing full-context layers through FlashAttention’s native dense-causal path will preserve the baseline attention pattern and quality while increasing throughput beyond 497M tokens, lowering val_bpb below 0.995558.

INTENDED_EDIT: Keep the SSSL architecture unchanged, but encode its full-context layers with FlashAttention’s `(-1, -1)` sentinel instead of an equivalent sliding-window configuration.

EVIDENCE: Native dense attention was valid and hardware-efficient at 38.44% MFU, while changing the frequency or size of attention windows reduced throughput and worsened val_bpb; isolating kernel dispatch avoids those architectural confounds.

<<<<<<< SEARCH
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        char_to_window = {"L": (-1, -1), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (-1, -1)
>>>>>>> REPLACE