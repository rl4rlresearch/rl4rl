MECHANISM: Single-microbatch, higher-update-frequency training

HYPOTHESIS: On the proven SSSL architecture, a 393,216-token batch executed as one 192-sequence microbatch will fit the available VRAM, reduce gradient accumulation, and provide more optimizer updates per token, lowering val_bpb below 0.995558.

INTENDED_EDIT: Restore the best-performing SSSL attention layout, increase device batch size to 192, and reduce total batch size to 393,216 so each optimizer step uses one microbatch.

EVIDENCE: Reference Design 1 achieved the best val_bpb, 0.995558, with SSSL while using only 45,060 MB peak VRAM; this leaves substantial H100 memory headroom for a larger microbatch and motivates testing more frequent updates without changing the proven model architecture or optimizer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven half-context layers; final layer is forced full-context
=======
WINDOW_PATTERN = "SSSL" # proven layout: two full-context anchor layers
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # 393K tokens; one 192-sequence microbatch per step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # use available H100 memory to eliminate accumulation
>>>>>>> REPLACE