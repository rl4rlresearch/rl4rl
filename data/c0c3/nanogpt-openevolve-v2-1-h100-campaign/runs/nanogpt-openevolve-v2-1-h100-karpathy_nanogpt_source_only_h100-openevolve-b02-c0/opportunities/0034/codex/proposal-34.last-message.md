MECHANISM: Tighter local attention for higher token throughput

HYPOTHESIS: Reducing the six short-window layers from 512-token to 256-token attention will lower val_bpb below 0.984293 by processing more training tokens while the two full-context layers retain global mixing.

INTENDED_EDIT: Change short attention from one-quarter to one-eighth of the 2048-token context; preserve the SSSL pattern, final full-context layer, optimizer, and schedules.

EVIDENCE: Cutting short attention from 1024 to 512 tokens improved val_bpb from 0.985730 to 0.984293 and increased processed tokens from 491.0M to 499.4M, directly motivating continuation along the successful window-size axis.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE