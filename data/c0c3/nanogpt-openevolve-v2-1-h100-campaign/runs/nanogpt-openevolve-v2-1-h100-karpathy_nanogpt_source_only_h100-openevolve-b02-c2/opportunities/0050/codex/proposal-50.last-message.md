MECHANISM: Post-global value-residual alignment

HYPOTHESIS: With the best verified 128/144 layer-4/layer-5 window allocation, relocating the layer-6 value embedding to layer 5 will lower val_bpb below 0.982455 by supplying token-specific values to the post-global transformation that benefited from additional context, while preserving parameter count.

INTENDED_EDIT: Restore the best 144/144/128/144/128/128 window schedule and change the eight-layer value-embedding placement from layers 2/4/6/8 to layers 2/4/5/8.

EVIDENCE: Moving eight context tokens from layer 4 to layer 5 improved val_bpb from 0.982662 to 0.982455, identifying layer 5 as the stronger post-global allocation target; relocating a value embedding to full-context layer 3 instead regressed to 0.983327, motivating alignment with the successful local layer rather than the global mixer.

<<<<<<< SEARCH
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2
=======
def has_ve(layer_idx, n_layer):
    """Place value embeddings at layers 2, 4, 5, and 8 in the tuned eight-layer model."""
    if n_layer == 8:
        return layer_idx in (1, 3, 4, 7)
    return layer_idx % 2 == (n_layer - 1) % 2
>>>>>>> REPLACE

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            15 * long_window // 256,
            19 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/120/152/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/128/144/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE