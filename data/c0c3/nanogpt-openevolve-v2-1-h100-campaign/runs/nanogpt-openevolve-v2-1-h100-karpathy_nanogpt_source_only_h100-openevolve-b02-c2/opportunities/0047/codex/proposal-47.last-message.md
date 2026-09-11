MECHANISM: Intermediate delayed post-global context allocation

HYPOTHESIS: A 12-token layer-4-to-layer-5 context transfer will lower val_bpb below 0.982455 by targeting the apparent optimum between the improving 8-token transfer and the slightly regressing 16-token transfer while preserving aggregate attention compute.

INTENDED_EDIT: Restore 128-token windows in layers 6 and 7, then use 124/148-token windows in layers 4/5 with full-context layers 3 and 8.

EVIDENCE: Transfers of 0, 8, and 16 tokens produced val_bpb values of 0.982662, 0.982455, and 0.982539 respectively, motivating a midpoint test between the two transferred schedules.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
            15 * long_window // 256,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            31 * long_window // 512,
            37 * long_window // 512,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144 pre-global; 144/136/128/120 post-global; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144 pre-global; 124/148/128/128 post-global; full context at layers 3 and 8
>>>>>>> REPLACE