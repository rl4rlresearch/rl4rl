MECHANISM: Sixteenth-context local attention

HYPOTHESIS: Reducing the six short-attention layers from 256 to 128 tokens while retaining both full-context layers will increase token throughput enough to lower val_bpb below 0.993215 within five minutes.

INTENDED_EDIT: Set short-attention windows to one-sixteenth of the 2048-token sequence length.

EVIDENCE: Reducing short windows from 512 to 256 increased processed tokens from 514.9M to 522.2M without degrading val_bpb, motivating one further measured reduction to locate the context-throughput optimum.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE