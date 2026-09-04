MECHANISM: Intermediate post-global attention compression

HYPOTHESIS: A 384-token layer-5 attention window will beat 0.983955 val_bpb by balancing the context lost at 128 tokens against the extra compute required at 512 tokens.

INTENDED_EDIT: Add a 384-token attention-window symbol and apply it only at layer 5; retain full attention at layers 4 and 8, 1024-token attention at layer 6, and 512-token attention elsewhere.

EVIDENCE: Layer 5 achieved 0.983955 at 256 tokens, outperforming 512 tokens at 0.984059, while further narrowing to 128 tokens regressed to 0.984297; 384 tokens is the untested aligned midpoint on the higher-context side.

<<<<<<< SEARCH
        assert all(c in "NSML" for c in pattern)
        long_window = config.sequence_len
        narrow_window = long_window // 8
        short_window = long_window // 4
        medium_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
            "N": (narrow_window, 0),
        }
=======
        assert all(c in "NTSML" for c in pattern)
        long_window = config.sequence_len
        narrow_window = long_window // 8
        three_sixteenths_window = 3 * long_window // 16
        short_window = long_window // 4
        medium_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
            "T": (three_sixteenths_window, 0),
            "N": (narrow_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLNMSL" # 256 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
=======
WINDOW_PATTERN = "SSSLTMSL" # 384 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
>>>>>>> REPLACE