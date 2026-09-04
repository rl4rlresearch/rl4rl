MECHANISM: Tighter local attention for higher token throughput

HYPOTHESIS: Reducing the six short-window layers from 256-token to 128-token attention will lower val_bpb below 0.983919 by processing more training tokens while the two full-context layers preserve global information flow.

INTENDED_EDIT: Change short attention from one-eighth to one-sixteenth of the 2048-token context, preserving the SSSL pattern, optimizer, and schedules.

EVIDENCE: Successive reductions from 1024 to 512 and then 256 tokens improved val_bpb from 0.985730 to 0.984293 and 0.983919, while increasing processed tokens from 491.0M to 499.4M and 512.0M; continuing this axis is the most informative next test.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE