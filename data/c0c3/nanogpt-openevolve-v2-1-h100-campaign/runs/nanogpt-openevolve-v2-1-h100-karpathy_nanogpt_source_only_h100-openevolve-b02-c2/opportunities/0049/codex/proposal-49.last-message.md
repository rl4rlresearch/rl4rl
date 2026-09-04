MECHANISM: Delayed pre-global context allocation

HYPOTHESIS: Transferring eight window tokens from layer 1 to layer 2 will lower val_bpb below 0.982455 by concentrating local context in the transformation immediately preceding the first full-context layer while preserving attention compute.

INTENDED_EDIT: Change the local-window schedule from 144/144/128/144/128/128 to 136/152/128/144/128/128, retaining full-context layers 3 and 8 and all optimizer settings.

EVIDENCE: The compute-matched eight-token transfer from layer 4 to layer 5 improved val_bpb from 0.982662 to 0.982455, showing that delaying context by one transformation can help; applying the same transfer to the two layers before the proven layer-3 global mixer tests whether that benefit generalizes across the other local stage.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            17 * long_window // 256,
            19 * long_window // 256,
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/128/144/128/128 local windows; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 136/152/128/144/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE