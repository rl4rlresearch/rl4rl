MECHANISM: Quarter-context local attention with periodic full-context layers

HYPOTHESIS: Reducing short attention windows from 1,024 to 512 tokens will increase token throughput enough to exceed 497M training tokens while periodic full-context layers preserve modeling quality, lowering val_bpb below 0.995558.

INTENDED_EDIT: Keep the successful batch size, learning rates, and architecture unchanged, but reduce each short sliding-attention window from half to one quarter of the 2,048-token context.

EVIDENCE: The single-microbatch experiment processed only 414.1M tokens versus the starting design’s 497.0M and regressed from 0.995558 to 1.001602 despite more optimizer steps, indicating token throughput is the stronger lever. Shorter local windows target attention cost without repeating the harmful batching change.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE