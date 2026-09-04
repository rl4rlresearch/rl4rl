MECHANISM: Targeted post-global context compression

HYPOTHESIS: Reducing only layer 4’s window from 512 to 256 tokens will lower val_bpb below 0.984083 by increasing throughput in a layer that did not benefit from wider context, while preserving the proven early 512-token, global, and layer-6 medium windows.

INTENDED_EDIT: Add an eighth-context window symbol and apply it only to layer 4, changing the pattern from SSSL​SLML to SSSL​NLML.

EVIDENCE: Widening layer 4 to 1024 tokens regressed val_bpb to 0.984595, while reducing every short layer to 256 regressed to 0.984865; a targeted reduction tests whether layer 4’s attention compute can be reclaimed without removing useful early-layer context.

<<<<<<< SEARCH
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 2
        short_window = long_window // 4
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
=======
        assert all(c in "NSML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 2
        short_window = long_window // 4
        narrow_window = long_window // 8
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
            "N": (narrow_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLML" # full context at 3,5,7; half context at 6
=======
WINDOW_PATTERN = "SSSLNLML" # full context at 3,5,7; quarter context at 4; half context at 6
>>>>>>> REPLACE