MECHANISM: Local batch-size optimum interpolation

HYPOTHESIS: A 112-sequence single-microbatch step will combine near-128-batch throughput with 14% more optimizer updates, achieving `val_bpb < 0.986636`.

INTENDED_EDIT: Increase both the global token batch and device batch from 96 to 112 sequences while retaining one microbatch per optimizer step.

EVIDENCE: Batch 96 achieved `val_bpb 0.986636`, narrowly outperforming batch 128 at `0.987111`, while batch 64 regressed to `0.991095`; these results place the observed optimum between 96 and 128 and motivate testing 112.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 96 * 2048 # ~197K tokens per optimizer step; one 96-sequence microbatch
=======
TOTAL_BATCH_SIZE = 112 * 2048 # ~229K tokens per optimizer step; one 112-sequence microbatch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 112  # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE