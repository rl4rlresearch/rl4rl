MECHANISM: Three-quarter-window layer-4 context interpolation

HYPOTHESIS: A 1536-token layer-4 window will beat 0.984059 val_bpb by preserving more of layer 4’s useful long-range context than the regressed 1024-token window while recovering some throughput from full attention.

INTENDED_EDIT: Add a 1536-token attention-window symbol and apply it only at layer 4; layer 6 remains at 1024 tokens, layer 8 remains full-attention, and all other layers remain at 512 tokens.

EVIDENCE: Reducing layer 4 directly from 2048 to 1024 tokens regressed val_bpb from 0.984059 to 0.985141 despite increasing total tokens from 500.2M to 502.5M, showing that layer 4 needs more context while leaving the untested 1536-token midpoint as the most direct context–throughput interpolation.

<<<<<<< SEARCH
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        medium_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
=======
        assert all(c in "SMHL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        medium_window = long_window // 2
        three_quarter_window = 3 * long_window // 4
        char_to_window = {
            "L": (long_window, 0),
            "H": (three_quarter_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSHSMSL" # 1536-token layer 4, 1024-token layer 6, full-attention layer 8
>>>>>>> REPLACE