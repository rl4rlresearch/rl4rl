MECHANISM: Eighth-context local attention with a final global anchor

HYPOTHESIS: Seven 256-token local-attention layers followed by one full-context layer will process at least 516.9M tokens and lower val_bpb below 0.993518.

INTENDED_EDIT: Restore the proven 524K-token batch, use a single forced final full-context anchor, and reduce all preceding local windows from 1024 to 256 tokens.

EVIDENCE: Moving from seven 512-token local layers to seven 384-token local layers improved val_bpb from 0.993870 to 0.993518 at comparable throughput, while removing the final global anchor worsened val_bpb to 1.001821; this motivates continuing the locality sweep while preserving that anchor.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # 262K tokens; one 128-sequence microbatch per step
=======
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
>>>>>>> REPLACE