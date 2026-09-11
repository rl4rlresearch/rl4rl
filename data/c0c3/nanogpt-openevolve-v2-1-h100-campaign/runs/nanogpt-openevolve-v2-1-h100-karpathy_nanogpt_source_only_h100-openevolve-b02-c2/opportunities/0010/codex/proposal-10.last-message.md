MECHANISM: Reduced short-horizon cautious decay

HYPOTHESIS: Halving Muon weight decay again from 0.1 to 0.05 will lower val_bpb below 0.985746 by reducing residual over-regularization during the short, high-update-count training run.

INTENDED_EDIT: Keep the best 262K-token design unchanged except for reducing cautious Muon weight decay to 0.05.

EVIDENCE: At the same 262K-token batch, reducing weight decay from 0.2 to 0.1 improved val_bpb from 0.987182 to 0.985746 while preserving roughly 490M-token throughput, directly motivating a controlled continuation of the observed trend.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.1      # normalize cumulative cautious decay for the higher update cadence
=======
WEIGHT_DECAY = 0.05     # reduce residual over-regularization in the short fixed-time run
>>>>>>> REPLACE