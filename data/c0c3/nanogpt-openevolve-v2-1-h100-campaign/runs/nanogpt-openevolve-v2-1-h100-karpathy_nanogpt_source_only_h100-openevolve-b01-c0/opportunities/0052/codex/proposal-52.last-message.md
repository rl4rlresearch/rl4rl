MECHANISM: Isolated post-global attention compression

HYPOTHESIS: Narrowing only layer 5 to 256-token attention will beat 0.984059 val_bpb by recovering throughput while preserving layer 7’s 512-token context.

INTENDED_EDIT: Add a 256-token attention window and use it only at layer 5; retain full attention at layers 4 and 8, 1024-token attention at layer 6, and 512-token attention elsewhere.

EVIDENCE: Narrowing layers 5 and 7 together reached 0.984190, only 0.000131 behind the best result; the previous attempt to isolate layer 5 was not verified, so this remains the direct unresolved attribution test.

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
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSLNMSL" # 256 at layer 5; 1024 at layer 6; full attention at layers 4 and 8
>>>>>>> REPLACE