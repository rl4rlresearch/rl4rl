MECHANISM: Near-full layer-4 context interpolation

HYPOTHESIS: A 1920-token layer-4 window will beat 0.984059 val_bpb by preserving nearly all useful global context while slightly reducing attention compute.

INTENDED_EDIT: Add a 1920-token near-long attention window and apply it only at layer 4; layer 6 remains at 1024 tokens, layer 8 remains full-attention, and all other layers remain at 512 tokens.

EVIDENCE: Layer-4 performance improved monotonically from 0.985141 at 1024 tokens to 0.984551 at 1536 and 0.984287 at 1792, approaching the 0.984059 full-attention result; 1920 is the untested midpoint between 1792 and 2048 that determines whether the optimum lies just below the endpoint.

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
        assert all(c in "SMLN" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 4
        medium_window = long_window // 2
        near_long_window = long_window * 15 // 16
        char_to_window = {
            "L": (long_window, 0),
            "N": (near_long_window, 0),
            "M": (medium_window, 0),
            "S": (short_window, 0),
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSMSL" # full attention at layers 4 and 8; 1024-token attention at layer 6
=======
WINDOW_PATTERN = "SSSNSMSL" # 1920-token attention at layer 4, 1024 at layer 6, full at layer 8
>>>>>>> REPLACE