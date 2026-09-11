MECHANISM: Concentrated medium-context late refinement

HYPOTHESIS: Giving only layer 6 a 1024-token attention window will reduce val_bpb below 0.984227 by retaining useful late-context refinement without the throughput cost of another full-context layer.

INTENDED_EDIT: Add a medium-window pattern symbol and change layer 6 from a 512-token local window to a 1024-token window, preserving the proven full-context layers at indices 3, 5, and 7.

EVIDENCE: Promoting layer 6 to full context nearly matched the winner at 0.984308 despite processing 2.8M fewer tokens, suggesting useful late-context capacity with excessive compute cost; widening all local layers regressed to 0.985663, motivating a concentrated intermediate window.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        pattern = config.window_pattern.upper()
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        medium_window = long_window // 2
        short_window = long_window // 4
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # three spaced full-context layers; S=quarter context
=======
WINDOW_PATTERN = "SSSLSLML" # full context at 3,5,7; half context at 6
>>>>>>> REPLACE