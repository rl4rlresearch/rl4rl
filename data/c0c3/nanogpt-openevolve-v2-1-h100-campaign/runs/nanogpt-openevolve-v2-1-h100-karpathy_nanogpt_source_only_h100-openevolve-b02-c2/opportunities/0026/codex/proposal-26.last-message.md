MECHANISM: Extreme local-window compression with preserved global integration

HYPOTHESIS: Reducing the six local-attention layers from 128 to 64 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.983766 by increasing throughput enough to offset the smaller local receptive field.

INTENDED_EDIT: Set short attention windows to one-thirty-second of the 2048-token context while preserving the proven SSSL pattern and optimizer settings.

EVIDENCE: Successive reductions from 1024 to 768, 512, 256, and 128 tokens monotonically improved val_bpb from 0.985746 to 0.984868, 0.984125, 0.983993, and 0.983766; testing 64 tokens directly probes whether that trend continues beyond the current best design.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 32
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 512-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 64-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE