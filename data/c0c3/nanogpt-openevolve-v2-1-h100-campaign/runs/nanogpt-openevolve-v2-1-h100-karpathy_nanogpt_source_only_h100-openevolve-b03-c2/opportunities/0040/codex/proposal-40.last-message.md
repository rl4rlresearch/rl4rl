MECHANISM: Quarter-context local attention for greater token exposure

HYPOTHESIS: Reducing short-layer attention from 1024 to 512 tokens while retaining two full-context layers and the best verified schedules will increase training throughput enough to achieve val_bpb below 0.985708.

INTENDED_EDIT: Restore the verified 35% AdamW cooldown and halve only the short attention window; keep the 59% Muon cooldown and all other settings unchanged.

EVIDENCE: The 35% AdamW design is best at 0.985708, while numerous subsequent schedule refinements regressed; this motivates testing compute allocation instead, preserving periodic full-context attention while spending less time on six local-attention layers.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.345  # locally interpolated AdamW cooldown near the observed optimum
=======
WARMDOWN_RATIO = 0.35   # best verified AdamW cooldown
>>>>>>> REPLACE