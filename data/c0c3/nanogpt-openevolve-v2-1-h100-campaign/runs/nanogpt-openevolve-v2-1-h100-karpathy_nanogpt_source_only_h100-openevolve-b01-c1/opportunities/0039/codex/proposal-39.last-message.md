MECHANISM: Pre-consolidation context-budget concentration

HYPOTHESIS: Reallocating layer 2’s marginal 256-token window increase to layer 3 will lower val_bpb below 0.983689 by providing 768-token context immediately before the first full-context consolidation layer without changing attention compute or throughput.

INTENDED_EDIT: Change the first-stage windows from 256/512/512/2048 to 256/256/768/2048 while retaining 256-token local windows and full-context layers elsewhere.

EVIDENCE: Widening layer 3 alone improved val_bpb from 0.984467 to 0.983718, whereas also widening layer 2 improved it by only 0.000029; concentrating the same added attention budget in the demonstrably valuable pre-consolidation layer directly tests whether proximity to global mixing matters more than distributed widening.

<<<<<<< SEARCH
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 4
        short_window = long_window // 8
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
=======
        assert all(c in "SMWL" for c in pattern)
        long_window = config.sequence_len
        wide_window = 3 * long_window // 8
        medium_window = long_window // 4
        short_window = long_window // 8
        char_to_window = {
            "L": (long_window, 0),
            "W": (wide_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SMMLSSSL" # concentrate both medium windows in the first consolidation stage
=======
WINDOW_PATTERN = "SSWLSSSL" # concentrate the same context budget immediately before first consolidation
>>>>>>> REPLACE