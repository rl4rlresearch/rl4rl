MECHANISM: Intermediate single-microbatch optimizer cadence

HYPOTHESIS: A 196,608-token batch will beat 0.985318 val_bpb by increasing optimizer updates over the 262,144-token design while retaining more throughput than the regressed 131,072-token design.

INTENDED_EDIT: Set both the total batch and device batch to 96 sequences, preserving one microbatch per optimizer step and the proven 512-token SSSL architecture.

EVIDENCE: The 262,144-token single-microbatch design achieved 0.985318 on 506.7M tokens, while 131,072 tokens increased update frequency but fell to 470.3M tokens and 0.991784; the midpoint tests the most plausible throughput–update-frequency tradeoff.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 96 * 2048 # 196,608 tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE