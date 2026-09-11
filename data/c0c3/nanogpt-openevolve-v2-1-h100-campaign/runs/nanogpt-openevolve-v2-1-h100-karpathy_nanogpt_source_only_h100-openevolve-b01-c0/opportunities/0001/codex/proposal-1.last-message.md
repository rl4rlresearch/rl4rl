MECHANISM: Quarter-context local attention with periodic global mixing

HYPOTHESIS: Reducing short attention windows from 1024 to 512 tokens will beat 0.995558 val_bpb by increasing token throughput while the full-attention layers at depths 4 and 8 preserve long-context integration.

INTENDED_EDIT: Set short-window layers to one quarter of the 2048-token context; retain the existing SSSL pattern and final full-attention layer.

EVIDENCE: The starting design reached 0.995558 val_bpb on 497.0M tokens at 39.58% MFU; six of eight layers currently spend attention compute on 1024-token windows despite global mixing occurring every fourth layer.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE