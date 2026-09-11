MECHANISM: Single-microbatch optimizer cadence

HYPOTHESIS: Halving the total batch to 262,144 tokens will beat 0.993324 val_bpb by nearly doubling optimizer updates while preserving the proven 128-sequence kernel shape and 512-token SSSL architecture.

INTENDED_EDIT: Make each existing 128-sequence device microbatch an optimizer step, eliminating two-way gradient accumulation without increasing VRAM.

EVIDENCE: The best design processed 514.3M tokens but made only 981 updates; the attempted 192-sequence single-microbatch design could not be verified, so this isolates higher update frequency without its larger-memory execution shape.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE