MECHANISM: Sixteenth-context local attention with a final global anchor

HYPOTHESIS: Seven 128-token local-attention layers followed by one full-context layer will process more than 525.3M tokens and lower val_bpb below 0.992854.

INTENDED_EDIT: Halve the seven local-attention windows from 256 to 128 tokens while preserving the proven final full-context anchor, architecture, batch, optimizer, and schedule.

EVIDENCE: Reducing local windows from 512 to 384 and then 256 tokens improved val_bpb from 0.993870 to 0.993518 to 0.992854; the 256-token design also reached the highest throughput at 525.3M tokens, while removing the final global anchor worsened val_bpb to 1.001821.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 16
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 128-token local layers, then one forced full-context anchor
>>>>>>> REPLACE