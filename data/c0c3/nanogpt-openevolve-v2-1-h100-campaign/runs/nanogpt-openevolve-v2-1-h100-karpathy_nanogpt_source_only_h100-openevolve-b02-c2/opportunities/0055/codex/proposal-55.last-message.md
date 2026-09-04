MECHANISM: Interpolated front-loaded pre-global context allocation

HYPOTHESIS: A four-token layer-2-to-layer-1 transfer will lower val_bpb below 0.982455 by targeting the apparent optimum between the verified 144/144 and 152/136 pre-global allocations while retaining the best 128/144 post-global allocation.

INTENDED_EDIT: Use 148/140/128/144/128/128 local windows with full-context attention at layers 3 and 8, preserving aggregate attention compute and all optimizer settings.

EVIDENCE: The 144/144 pre-global allocation achieved 0.982455, while shifting eight tokens forward to 152/136 scored a close 0.982489 and shifting eight backward to 136/152 regressed to 0.982746; this asymmetric response motivates testing the four-token midpoint in the favorable direction.

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
            37 * long_window // 512,
            35 * long_window // 512,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/120/152/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 148/140/128/144/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE