MECHANISM: Finer-grained attention heads at fixed model width

HYPOTHESIS: Replacing four 128-dimensional attention heads with eight 64-dimensional heads, while holding depth, width, parameter count, window pattern, and batching effectively constant, will improve attention specialization and reduce val_bpb below 0.995558 without materially reducing training tokens.

INTENDED_EDIT: Set HEAD_DIM to 64; the model remains 512-dimensional but uses eight attention heads instead of four.

EVIDENCE: Changing depth or attention-window topology reduced throughput and regressed val_bpb, while the 8-layer SSSL baseline reached 0.995558. This isolates an untested attention-capacity tradeoff without repeating those changes.

<<<<<<< SEARCH
HEAD_DIM = 128          # target head dimension for attention
=======
HEAD_DIM = 64           # finer-grained attention heads at unchanged model width
>>>>>>> REPLACE