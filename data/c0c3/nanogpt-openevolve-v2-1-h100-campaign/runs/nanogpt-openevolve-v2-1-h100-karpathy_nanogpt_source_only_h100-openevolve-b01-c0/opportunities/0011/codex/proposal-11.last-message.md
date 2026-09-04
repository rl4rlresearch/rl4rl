MECHANISM: Intermediate 229K-token single-microbatch optimizer cadence

HYPOTHESIS: A 229,376-token batch will beat 0.985318 val_bpb by gaining update frequency over the 262,144-token design while retaining more throughput than the nearly competitive 196,608-token design.

INTENDED_EDIT: Set both the total batch and device batch to 112 sequences, preserving one microbatch per optimizer step and the proven 512-token SSSL architecture.

EVIDENCE: The 262,144-token design achieved 0.985318 on 506.7M tokens, while 196,608 tokens achieved a close 0.986155 on 492.9M tokens with 30% more updates; testing their midpoint isolates the most plausible remaining throughput–update-frequency optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 229376 # 112 sequences, one microbatch per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 112  # per-device batch size (one optimizer-step microbatch)
>>>>>>> REPLACE