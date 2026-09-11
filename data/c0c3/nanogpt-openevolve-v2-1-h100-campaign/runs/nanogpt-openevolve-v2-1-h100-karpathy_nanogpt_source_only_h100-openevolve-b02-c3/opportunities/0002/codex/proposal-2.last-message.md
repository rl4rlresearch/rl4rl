MECHANISM: More aggressive local-attention compression

HYPOTHESIS: Reducing the six short-attention layers from 512 to 256 tokens will raise throughput beyond 513.3M tokens while the two full-context layers retain enough global modeling to beat 0.993287 val_bpb.

INTENDED_EDIT: Set short attention to one-eighth of the 2048-token sequence, preserving the SSSL pattern and forced full-context final layer.

EVIDENCE: Reducing short windows from 1024 to 512 increased training tokens from 497.0M to 513.3M and improved val_bpb from 0.995558 to 0.993287, motivating the next factor-of-two test.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=eighth context
>>>>>>> REPLACE