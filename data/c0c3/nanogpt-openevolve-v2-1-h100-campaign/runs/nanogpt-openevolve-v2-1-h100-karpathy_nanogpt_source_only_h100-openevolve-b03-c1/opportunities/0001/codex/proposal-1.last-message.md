MECHANISM: Denser periodic global attention with cheaper local layers

HYPOTHESIS: Reducing short attention windows from 1024 to 512 tokens will increase training throughput enough to lower val_bpb below 0.995558, while the full-context layers at indices 3 and 7 preserve long-range information.

INTENDED_EDIT: Use quarter-context windows for “S” layers while retaining the existing SSSL pattern and full-context final layer.

EVIDENCE: The baseline reaches val_bpb 0.995558 after 497.0M tokens at 39.58% MFU; attention remains a substantial compute cost, and six of eight layers currently use relatively expensive half-context windows.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE