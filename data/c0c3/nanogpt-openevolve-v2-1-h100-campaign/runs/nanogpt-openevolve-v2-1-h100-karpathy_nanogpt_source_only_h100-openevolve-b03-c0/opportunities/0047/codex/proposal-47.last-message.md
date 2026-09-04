MECHANISM: Mild early-global context compression

HYPOTHESIS: Reducing the first global layer’s window to 1,792 tokens will achieve `val_bpb < 0.981072` by reallocating compute while retaining substantially more context than the harmful 512-token final-layer setting.

INTENDED_EDIT: Reduce only the first full-context layer from 2,048 to 1,792 tokens; retain the 512-token first pre-global window, 128-token local windows, and 1,152-token final window.

EVIDENCE: Compressing the final global layer ultimately improved `val_bpb` from `0.982068` at 2,048 tokens to `0.981072` at 1,152, while the aggressive 512-token result regressed to `0.985963`; this motivates a conservative compression test on the remaining full-context layer.

<<<<<<< SEARCH
            if char == "L":
                window = long_window
            else:
=======
            if char == "L":
                is_first_global = layer_idx < config.n_layer // 2
                window = long_window * 7 // 8 if is_first_global else long_window
            else:
>>>>>>> REPLACE