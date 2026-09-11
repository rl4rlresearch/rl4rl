MECHANISM: Hierarchical local, mid-range, and global attention

HYPOTHESIS: Replacing the intermediate 2048-token global layer with a 1024-token window will retain enough cross-window mixing to beat 0.993365 val_bpb while recovering some of the throughput gained by removing that global layer entirely.

INTENDED_EDIT: Add a medium-window attention type and use SSSM repetition, yielding 512-token local layers, a 1024-token layer at index 3, and mandatory full-context attention at the final layer.

EVIDENCE: The two-global-layer design achieved 0.993365, while replacing its intermediate global layer with 512-token attention increased tokens from 512.2M to 516.9M but regressed to 0.994122; a 1024-token intermediate layer directly tests the quality-throughput midpoint.

<<<<<<< SEARCH
    window_pattern: str = "SSSL"
=======
    window_pattern: str = "SSSM"
>>>>>>> REPLACE

<<<<<<< SEARCH
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
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
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSM" # L=full, M=half context, S=quarter context
>>>>>>> REPLACE