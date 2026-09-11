MECHANISM: Context-favoring five-eighths sliding attention

HYPOTHESIS: Increasing local-attention windows from 1,024 to 1,280 tokens will recover more useful context than the modest throughput reduction costs, lowering val_bpb below 0.984068.

INTENDED_EDIT: Set every `S` attention window to five-eighths of the 2,048-token sequence length.

EVIDENCE: Reducing `S` windows to 768 increased total tokens from 478.5M to 481.6M but worsened val_bpb from 0.984068 to 0.984407, indicating that additional context was more valuable than the measured throughput gain and motivating a test in the opposite direction.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 5 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=five-eighths context
>>>>>>> REPLACE