MECHANISM: Eighth-context local attention

HYPOTHESIS: Reducing the six short-attention layers from 512 to 256 tokens while retaining both full-context layers will increase throughput enough to achieve val_bpb below 0.993219 within five minutes.

INTENDED_EDIT: Set short-attention windows to one-eighth of the 2048-token sequence length.

EVIDENCE: Reducing short windows from 1024 to 512 increased processed tokens from 497.0M to 514.9M and improved val_bpb from 0.995558 to 0.993219, motivating a further measured reduction.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE