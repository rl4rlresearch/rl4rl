MECHANISM: Further update-count-normalized cautious weight decay

HYPOTHESIS: Halving Muon weight decay from 0.1 to 0.05 will reduce val_bpb below 0.986664 by further limiting cumulative parameter shrinkage while preserving throughput.

INTENDED_EDIT: Reduce the initial cautious Muon weight decay to 0.05; retain its existing linear decay schedule and all other settings.

EVIDENCE: Halving Muon weight decay from 0.2 to 0.1 improved val_bpb from 0.987174 to 0.986664 without materially changing throughput, making another halving the most informative continuation of the only recently successful optimization axis.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.1      # preserve cumulative decay horizon after doubling optimizer updates
=======
WEIGHT_DECAY = 0.05     # further reduce cumulative cautious decay at the doubled update frequency
>>>>>>> REPLACE