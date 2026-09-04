MECHANISM: Conservative delayed post-global context allocation

HYPOTHESIS: A four-token layer-4-to-layer-5 transfer will lower val_bpb below 0.982455 by refining the favorable interval between the improving eight-token transfer and the untransferred baseline while preserving aggregate attention compute.

INTENDED_EDIT: Restore 144-token windows in layers 1 and 2, then use 132/140-token windows in layers 4/5 with 128-token windows in layers 6/7 and full context at layers 3/8.

EVIDENCE: Moving eight tokens from layer 4 to layer 5 improved val_bpb from 0.982662 to 0.982455, while larger 12- and 16-token transfers regressed to 0.982730 and 0.982539; a four-token transfer tests the unexplored lower side of the apparent optimum.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
            19 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            33 * long_window // 512,
            35 * long_window // 512,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/136/128/152/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/132/140/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE