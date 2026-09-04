MECHANISM: Delayed post-global context allocation

HYPOTHESIS: Moving eight local-window tokens from layer 4 to layer 5 will lower val_bpb below 0.982662 by reducing context in the layer where isolated expansion regressed while strengthening the following transformation, without changing attention compute.

INTENDED_EDIT: Change the local-window schedule from 144/144/136/136/128/128 to 144/144/128/144/128/128, retaining full-context layers 3 and 8 and all optimizer settings.

EVIDENCE: Expanding layer 4 alone from 136 to 144 worsened val_bpb from 0.982662 to 0.982881, while the compute-matched layer-4 expansion plus layer-7 pruning scored 0.982945; this motivates reallocating context away from layer 4 rather than adding more there.

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
            long_window // 16,
            9 * long_window // 128,
            long_window // 16,
            long_window // 16,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSLSSSSL" # 144/144/136/136/128/128 local taper; full context at layers 3 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/128/144/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE