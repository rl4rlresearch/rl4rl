MECHANISM: Quarter-context local attention with periodic global layers

HYPOTHESIS: Reducing short-layer attention from 1024 to 512 tokens will lower val_bpb by increasing tokens processed within five minutes, while the two full-context layers preserve long-range modeling capacity.

INTENDED_EDIT: Set sliding-window layers to one-quarter of the 2048-token sequence length.

EVIDENCE: The starting design reaches val_bpb 0.995558 on 497.0M tokens at 39.58% MFU; shortening six of eight attention windows targets additional throughput without changing parameters or optimization.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE