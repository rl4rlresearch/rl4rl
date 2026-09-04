MECHANISM: Native dense causal attention on every layer

HYPOTHESIS: Replacing sliding-window attention with FlashAttention’s native full-causal path will avoid the hardware-efficiency loss seen when windows were shortened and improve context modeling enough to reduce val_bpb below 0.995558.

INTENDED_EDIT: Use full-context attention in all layers and represent it with FlashAttention’s optimized `(-1, -1)` window sentinel.

EVIDENCE: Reducing the short window from 1,024 to 512 decreased throughput from 497.0M to 390.1M tokens and worsened val_bpb to 1.011628, indicating narrower local attention is counterproductive on this workload.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "L"    # native full-context causal attention in every layer
>>>>>>> REPLACE

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