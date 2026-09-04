MECHANISM: Amplified delayed post-global context allocation

HYPOTHESIS: Doubling the verified layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by further concentrating local attention in the second transformation after global mixing while preserving total attention compute.

INTENDED_EDIT: Use 144/144/120/152/128/128 local windows with full-context layers 3 and 8, transferring another eight tokens from layer 4 to layer 5.

EVIDENCE: The compute-matched shift from 144/144/136/136/128/128 to 144/144/128/144/128/128 improved val_bpb from 0.982662 to 0.982455, while expanding layer 4 alone regressed to 0.982881; this motivates a dose-response test of the same transfer direction.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            15 * long_window // 256,
            19 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/144/136/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/120/152/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE