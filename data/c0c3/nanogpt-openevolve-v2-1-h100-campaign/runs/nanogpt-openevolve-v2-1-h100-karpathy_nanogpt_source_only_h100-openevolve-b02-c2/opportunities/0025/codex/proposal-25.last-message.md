MECHANISM: Aggressive local-window compression with preserved global integration

HYPOTHESIS: Reducing the six local-attention layers from 256 to 128 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.983993 by increasing token throughput enough to offset the reduced local receptive field.

INTENDED_EDIT: Change short attention windows from one-eighth to one-sixteenth of the 2048-token context, preserving the SSSL pattern and all optimizer settings.

EVIDENCE: Successive reductions from 1024 to 768, 512, and 256 tokens improved val_bpb from 0.985746 to 0.984868, 0.984125, and 0.983993 while increasing throughput; testing 128 tokens extends this monotonic compression series and probes its likely boundary.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 256-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 128-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE