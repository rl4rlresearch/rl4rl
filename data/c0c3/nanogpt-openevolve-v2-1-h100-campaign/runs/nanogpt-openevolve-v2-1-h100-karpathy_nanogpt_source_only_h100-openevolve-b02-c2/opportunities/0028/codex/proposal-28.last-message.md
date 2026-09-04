MECHANISM: Quadratic-bracket local-window refinement

HYPOTHESIS: Using 144-token local windows will lower val_bpb below 0.983766 by adding context to the best 128-token design while remaining near its throughput, consistent with the measured minimum implied by the 64/128/192-token bracket.

INTENDED_EDIT: Set the six short-attention layers to 144 tokens while preserving full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: Local windows of 64, 128, and 192 tokens produced val_bpb values of 0.984784, 0.983766, and 0.984182; quadratic interpolation of this asymmetric bracket places the estimated minimum near 141 tokens, motivating an aligned 144-token test.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        long_window = config.sequence_len
        short_window = 9 * long_window // 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # 144-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE