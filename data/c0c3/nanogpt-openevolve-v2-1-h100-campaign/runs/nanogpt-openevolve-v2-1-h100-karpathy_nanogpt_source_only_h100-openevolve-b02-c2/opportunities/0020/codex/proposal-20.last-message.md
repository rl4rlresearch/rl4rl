MECHANISM: Single global-attention consolidation

HYPOTHESIS: Keeping only the final layer full-context will lower val_bpb below 0.985746 by increasing token throughput while preserving one layer that can integrate the entire sequence.

INTENDED_EDIT: Restore the best verified Muon weight decay of 0.1 and change the attention pattern from two full-context layers to one final full-context layer.

EVIDENCE: Weight decay 0.1 with the 50% linear warmdown achieved the best val_bpb of 0.985746, while all tested decay and warmdown refinements were worse; this motivates restoring that baseline and testing compute reallocation through the untested window pattern.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSS" # use half context except for the always-full final layer
>>>>>>> REPLACE

<<<<<<< SEARCH
WEIGHT_DECAY = 0.125    # interpolate near the observed cautious-decay optimum
=======
WEIGHT_DECAY = 0.1      # best verified cautious-decay setting
>>>>>>> REPLACE