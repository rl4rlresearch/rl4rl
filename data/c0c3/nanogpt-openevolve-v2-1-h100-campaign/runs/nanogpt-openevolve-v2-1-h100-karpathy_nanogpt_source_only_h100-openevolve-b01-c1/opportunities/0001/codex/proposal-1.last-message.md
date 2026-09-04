MECHANISM: Quarter-context local attention

HYPOTHESIS: Reducing short-layer attention from 1024 to 512 tokens while retaining full-context layers will increase token throughput enough to lower val_bpb below 0.995558 within five minutes.

INTENDED_EDIT: Use quarter-context windows for the six short-attention layers; keep the existing SSSL pattern and mandatory full-context final layer.

EVIDENCE: The baseline processes 497.0M tokens at 39.58% MFU, indicating substantial opportunity to trade excess short-layer attention compute for additional training tokens.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE