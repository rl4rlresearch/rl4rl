MECHANISM: Front-loaded pre-global context allocation

HYPOTHESIS: With the best verified 128/144 layer-4/layer-5 allocation restored, transferring eight window tokens from layer 2 to layer 1 will lower val_bpb below 0.982455 by favoring initial context formation before the layer-3 global mixer.

INTENDED_EDIT: Use 152/136/128/144/128/128 local windows with full-context layers 3 and 8, preserving aggregate attention compute and all optimizer settings.

EVIDENCE: The best 144/144/128/144/128/128 schedule achieved 0.982455, while transferring eight pre-global tokens in the opposite direction to produce 136/152 regressed to 0.982746; this motivates testing the reverse allocation while retaining the proven post-global windows.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            31 * long_window // 512,
            37 * long_window // 512,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            19 * long_window // 256,
            17 * long_window // 256,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144 pre-global; 124/148/128/128 post-global; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 152/136 pre-global; 128/144/128/128 post-global; full context at layers 3 and 8
>>>>>>> REPLACE