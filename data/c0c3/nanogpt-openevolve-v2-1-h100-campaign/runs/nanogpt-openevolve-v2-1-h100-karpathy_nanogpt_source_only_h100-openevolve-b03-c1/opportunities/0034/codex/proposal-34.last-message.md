MECHANISM: Compute reallocation from local context to token throughput

HYPOTHESIS: Halving short-attention windows to 256 tokens while retaining full-context layers at indices 3, 5, and 7 will reduce val_bpb below 0.984227 by increasing processed tokens without removing proven global mixing.

INTENDED_EDIT: Change the five short-context layers from quarter-context (512 tokens) to eighth-context (256 tokens), preserving all other architecture and training settings.

EVIDENCE: Widening short windows to 768 tokens regressed val_bpb to 0.985663, while adding global mixing improved val_bpb despite lower throughput; this suggests long-range capacity is best concentrated in the three global layers and local-attention compute can be reduced.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
=======
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=eighth context
>>>>>>> REPLACE

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE