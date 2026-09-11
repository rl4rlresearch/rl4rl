MECHANISM: Midpoint optimizer-batch refinement

HYPOTHESIS: A 245,760-token optimizer batch will retain roughly 520M training tokens while lowering val_bpb below 0.985657.

INTENDED_EDIT: Use one 120-sequence microbatch per optimizer step, preserving the proven architecture, learning rates, optimizer, and 60% linear warmdown.

EVIDENCE: The 262K batch achieved 0.985657 and the nearby 229K batch achieved 0.985754, both outperforming the 196K batch at 0.986505; testing their midpoint targets the narrow empirical optimum.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens; one 128-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 15 * 2**14 # ~246K tokens; one 120-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # efficient per-device batch size
=======
DEVICE_BATCH_SIZE = 120  # one microbatch per 245,760-token optimizer step
>>>>>>> REPLACE