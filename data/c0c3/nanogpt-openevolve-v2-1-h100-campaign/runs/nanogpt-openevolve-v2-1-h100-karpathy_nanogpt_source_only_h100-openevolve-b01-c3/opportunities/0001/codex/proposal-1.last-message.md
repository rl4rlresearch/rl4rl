MECHANISM: Higher optimizer-update density via smaller global batch

HYPOTHESIS: Halving the global batch will increase optimizer steps per token enough to reduce val_bpb below 0.995558 within five minutes, despite modest optimizer overhead.

INTENDED_EDIT: Reduce total batch size from 524,288 to 262,144 tokens while retaining the efficient 128-sequence device batch, changing gradient accumulation from two microbatches to one.

EVIDENCE: The starting design processes 497.0M tokens but performs only 948 updates, suggesting optimization may be update-limited rather than data-limited.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE