MECHANISM: Fine-grained local-window interpolation

HYPOTHESIS: Using 136-token local windows will lower val_bpb below 0.983758 by moving toward the approximately 138-token minimum implied by the measured 128/144/192-token bracket while slightly reducing attention compute.

INTENDED_EDIT: Reduce all six short-attention layers from 144 to 136 tokens, preserving full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: Windows of 128, 144, and 192 tokens achieved val_bpb values of 0.983766, 0.983758, and 0.984182; quadratic interpolation of this tighter bracket places its minimum near 138 tokens, motivating the nearest 8-token-aligned setting.

<<<<<<< SEARCH
        short_window = 9 * long_window // 128
=======
        short_window = 17 * long_window // 256
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 136-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE