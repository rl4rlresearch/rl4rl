MECHANISM: Near-full layer-4 context interpolation

HYPOTHESIS: A 1792-token layer-4 window will beat 0.984059 val_bpb by retaining nearly all useful global context while recovering some throughput from full attention.

INTENDED_EDIT: Add a seven-eighths attention-window symbol and apply it only at layer 4; layer 6 remains at 1024 tokens, layer 8 remains full-attention, and all other layers remain at 512 tokens.

EVIDENCE: Reducing layer 4 from 2048 to 1024 regressed val_bpb to 0.985141, while 1536 improved that result to 0.984551; the monotonic recovery toward full attention motivates testing the remaining midpoint between 1536 and 2048.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
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
        pattern = config.window_pattern.upper()
        assert all(c in "SMXL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        medium_window = long_window // 2
        extra_long_window = long_window * 7 // 8
        char_to_window = {
            "L": (long_window, 0),
            "X": (extra_long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSXSMSL" # 1792-token attention at layer 4, 1024 at layer 6, full attention at layer 8
>>>>>>> REPLACE