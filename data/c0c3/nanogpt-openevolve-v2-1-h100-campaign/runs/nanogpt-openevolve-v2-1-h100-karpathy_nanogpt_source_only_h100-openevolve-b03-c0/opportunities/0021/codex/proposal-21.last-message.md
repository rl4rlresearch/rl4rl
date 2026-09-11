MECHANISM: Higher update density via smaller one-microbatch batches

HYPOTHESIS: An 80-sequence batch will complete more than 2,518 optimizer updates and achieve `val_bpb < 0.985780` by improving optimization frequency despite modestly lower token throughput.

INTENDED_EDIT: Reduce both total and device batch size from 96 to 80 sequences, preserving single-microbatch updates and the current best architecture, softcap, optimizer, and schedule.

EVIDENCE: The evidence identifies the 96-sequence design’s denser updates as beneficial, while multiple schedule-only changes failed to improve it; an incremental batch reduction directly tests whether additional update density extends that gain.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 96 * 2048 # ~197K tokens per optimizer step; one 96-sequence microbatch
=======
TOTAL_BATCH_SIZE = 80 * 2048 # ~164K tokens per optimizer step; one 80-sequence microbatch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 80   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE