MECHANISM: Replace redundant intermediate global attention with local attention

HYPOTHESIS: Using seven 384-token local-attention layers and only the forced final full-context layer will lower val_bpb below 0.983803 by increasing training throughput while stacked local layers retain a sequence-wide receptive field.

INTENDED_EDIT: Change the attention pattern from six short and two full-context layers to seven short layers followed by the existing forced full-context final layer.

EVIDENCE: Increasing the short window from 384 to 448 tokens reduced training from 508.8M to 499.4M tokens and regressed val_bpb from 0.983803 to 0.984134, while windows below 384 also regressed. This motivates preserving the best 384-token window and seeking throughput by removing one expensive full-context layer instead.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "S"    # seven local layers; final layer is forcibly full-context
>>>>>>> REPLACE