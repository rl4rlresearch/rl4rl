MECHANISM: Intermediate layer-6 attention span

HYPOTHESIS: Giving layer 6 a 1024-token window will beat 0.984244 val_bpb by retaining much of the improvement from making that layer full-attention while recovering throughput toward the two-full-layer design.

INTENDED_EDIT: Add a medium-window pattern symbol and use it at layer 6; layers 4 and 8 remain full-attention, while all other layers retain 512-token windows.

EVIDENCE: Expanding layer 6 from 512 to 2048 improved val_bpb from 0.984313 to 0.984244 despite reducing training tokens from 506.2M to 499.6M, motivating a direct context-versus-throughput interpolation.

<<<<<<< SEARCH
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        medium_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
>>>>>>> REPLACE