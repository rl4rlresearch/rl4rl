MECHANISM: Quarter-context local attention with full-context anchors

HYPOTHESIS: Reducing S layers from 1024-token to 512-token windows while restoring the proven 524K-token batch will process more than 497M tokens, and the two full-context anchor layers will preserve enough global information to lower val_bpb below 0.995558.

INTENDED_EDIT: Use quarter-context windows for the six local-attention layers and restore Reference Design 1’s 128-sequence microbatch with two-step gradient accumulation.

EVIDENCE: The SSSL baseline achieved 0.995558 on 497.0M tokens, whereas making every layer full-context reduced throughput to 403.7M tokens and worsened val_bpb to 1.011302; this motivates testing more aggressive locality while retaining full-context anchors.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens; one 192-sequence microbatch per step
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens; two 128-sequence microbatches per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 192  # use available H100 memory to eliminate accumulation
=======
DEVICE_BATCH_SIZE = 128  # efficient per-device batch size
>>>>>>> REPLACE