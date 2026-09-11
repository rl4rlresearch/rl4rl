MECHANISM: Post-global local-context restart

HYPOTHESIS: With full-context layers 3 and 8, shifting eight local-window tokens from layer 7 to layer 4 will lower val_bpb below 0.982662 by strengthening the first transformation after global mixing without changing aggregate attention compute.

INTENDED_EDIT: Place full-context attention at layers 3 and 8 and use 144/144 local windows before the first global layer followed by a 144/136/128/120 taper.

EVIDENCE: Moving the first full-context layer from 4 to 3 improved val_bpb from 0.983152 to 0.982662. Separately, moving a larger window toward the terminal local layer regressed to 0.983660, motivating compute reallocation toward the first post-global layer.

<<<<<<< SEARCH
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            17 * long_window // 256,
            long_window // 16,
            long_window // 16,
        )
=======
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
            15 * long_window // 256,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144/144/136/136/128/128 local taper; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144 pre-global; 144/136/128/120 post-global; full context at layers 3 and 8
>>>>>>> REPLACE