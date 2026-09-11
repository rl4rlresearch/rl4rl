MECHANISM: Denser local attention with periodic global layers

HYPOTHESIS: Reducing the six short-attention layers from 1024 to 512 tokens will increase throughput beyond 497M tokens while the two full-context layers preserve enough long-range modeling to beat 0.995558 val_bpb.

INTENDED_EDIT: Change short attention windows from half-context to quarter-context, retaining the existing SSSL pattern and forced full-context final layer.

EVIDENCE: The starting design reaches 0.995558 val_bpb at 497M tokens and 39.58% MFU; attention-window compute is therefore a direct opportunity to train on more data within the fixed 300-second window.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE