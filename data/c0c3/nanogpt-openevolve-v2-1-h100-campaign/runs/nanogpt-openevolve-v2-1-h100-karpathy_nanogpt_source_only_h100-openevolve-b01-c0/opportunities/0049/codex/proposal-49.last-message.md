MECHANISM: Concentrated mid-stack attention reallocation

HYPOTHESIS: Narrowing layers 5 and 7 to 256-token attention while retaining 1024 tokens at layer 6 will beat 0.984059 val_bpb by recovering throughput without removing the context expansion proven most useful at layer 6.

INTENDED_EDIT: Add a 256-token narrow-window symbol and apply it at layers 5 and 7; preserve full attention at layers 4 and 8, 1024-token attention at layer 6, and 512-token attention at layers 1–3.

EVIDENCE: Expanding layer 6 from 512 to 1024 improved val_bpb from 0.984313 to 0.984059 despite reducing training tokens from 506.2M to 500.2M, while adding 1024-token attention at layer 2 regressed to 0.984559. This motivates reallocating attention compute toward layer 6 rather than distributing it across adjacent layers.

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
WINDOW_PATTERN = "SSSLNMNL" # full at 4/8, 1024-token at 6, 256-token at 5/7
>>>>>>> REPLACE