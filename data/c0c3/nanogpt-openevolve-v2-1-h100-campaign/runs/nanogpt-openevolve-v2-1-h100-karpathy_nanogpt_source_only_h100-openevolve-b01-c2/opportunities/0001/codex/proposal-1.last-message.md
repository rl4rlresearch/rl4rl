MECHANISM: Quarter-context sliding attention with periodic full-context layers

HYPOTHESIS: Reducing short-layer attention windows from 1024 to 512 tokens will increase tokens processed within five minutes enough to beat the current 0.995558 val_bpb, while the two full-context layers preserve long-range modeling.

INTENDED_EDIT: Change the six short-window layers to use one-quarter context; retain the existing SSSL pattern and forced full-context final layer.

EVIDENCE: The current depth-8 design reaches 0.995558 val_bpb on 497.0M tokens at 39.58% MFU, so reducing attention work is a targeted way to train on more data without reducing model parameters or depth.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE