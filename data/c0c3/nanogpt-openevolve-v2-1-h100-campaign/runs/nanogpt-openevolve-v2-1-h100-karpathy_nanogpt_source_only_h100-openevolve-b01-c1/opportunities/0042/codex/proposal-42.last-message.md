MECHANISM: Partial front-loaded context widening

HYPOTHESIS: Using a 384-token window in layer 2 while retaining layer 3’s 512-token window will lower val_bpb below 0.983689 by preserving useful early context while recovering throughput from layer 2’s marginal widening.

INTENDED_EDIT: Add a 384-token transition window and change the schedule from SMMLSSSL to STMLSSSL.

EVIDENCE: Widening layer 3 alone reached 0.983718 at 513.0M tokens, while also widening layer 2 improved only to 0.983689 and reduced training to 511.4M tokens; interpolating layer 2’s window tests whether its small contextual benefit can be retained with less throughput loss.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
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
        pattern = config.window_pattern.upper()
        assert all(c in "STML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 4
        transition_window = 3 * long_window // 16
        short_window = long_window // 8
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "T": (transition_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SMMLSSSL" # concentrate both medium windows in the first consolidation stage
=======
WINDOW_PATTERN = "STMLSSSL" # partially widen layer 2 and fully widen layer 3 before consolidation
>>>>>>> REPLACE