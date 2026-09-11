MECHANISM: Moderately enlarged single-microbatch training

HYPOTHESIS: A 294,912-token batch will beat 0.985318 val_bpb by improving device utilization and token throughput, while remaining safely below the unverified 393,216-token configuration.

INTENDED_EDIT: Increase both total and device batch size from 128 to 144 sequences, preserving one optimizer update per microbatch and the proven 512-token SSSL architecture.

EVIDENCE: Increasing the single-microbatch size from 96 to 112 to 128 sequences monotonically improved val_bpb from 0.986155 to 0.985719 to 0.985318 and raised throughput; 144 is the nearest upward test before the failed 192-sequence attempt.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 144 * 2048 # ~295K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 144  # per-device batch size (one microbatch per optimizer step)
>>>>>>> REPLACE