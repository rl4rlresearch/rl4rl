MECHANISM: Deeper post-global attention compression

HYPOTHESIS: Reducing only layer 5 from 256-token to 128-token attention will beat 0.983955 val_bpb by increasing throughput while preserving the 512-token context at layer 7.

INTENDED_EDIT: Add a 128-token attention-window symbol and apply it only at layer 5; retain full attention at layers 4 and 8, 1024-token attention at layer 6, and 512-token attention elsewhere.

EVIDENCE: Narrowing only layer 5 from 512 to 256 tokens improved val_bpb from 0.984059 to 0.983955 and increased training tokens from 500.2M to 503.6M, while narrowing layer 7 as well regressed to 0.984190.

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
        assert all(c in "TNSML" for c in pattern)
        long_window = config.sequence_len
        tiny_window = long_window // 16
        narrow_window = long_window // 8
        short_window = long_window // 4
        medium_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
            "N": (narrow_window, 0),
            "T": (tiny_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLNMSL" # 256 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
=======
WINDOW_PATTERN = "SSSLTMSL" # 128 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
>>>>>>> REPLACE