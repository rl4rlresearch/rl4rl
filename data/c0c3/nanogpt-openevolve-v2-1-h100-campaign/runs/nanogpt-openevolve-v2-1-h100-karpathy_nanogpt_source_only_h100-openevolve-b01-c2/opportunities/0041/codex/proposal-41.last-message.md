MECHANISM: Tight lower-bracket optimizer-batch refinement

HYPOTHESIS: A 256,000-token optimizer batch will retain roughly 520M training tokens while lowering val_bpb below 0.985181.

INTENDED_EDIT: Use one 125-sequence microbatch per optimizer step while preserving the architecture, optimizer hyperparameters, and 60% linear warmdown.

EVIDENCE: The 258,048-token batch achieved 0.985181 versus 0.985215 at 253,952 tokens, while 260,096 tokens regressed to 0.986845; testing 256,000 tokens fills the remaining midpoint in the strongest lower bracket without relying on the anomalous 127-sequence throughput.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 125 * 2**11 # 256K tokens; one 125-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
=======
DEVICE_BATCH_SIZE = 125  # one microbatch per 256,000-token optimizer step
>>>>>>> REPLACE