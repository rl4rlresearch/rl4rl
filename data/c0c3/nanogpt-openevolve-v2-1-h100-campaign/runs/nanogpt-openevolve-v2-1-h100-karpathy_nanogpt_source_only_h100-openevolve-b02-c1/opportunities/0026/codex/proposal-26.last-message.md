MECHANISM: Update-count-normalized Muon weight decay

HYPOTHESIS: Reducing Muon weight decay to 0.14 will preserve approximately the 256K-batch design’s cumulative decay while retaining the best 176K update frequency, lowering val_bpb below 0.986967.

INTENDED_EDIT: Reduce cautious Muon weight decay from 0.20 to 0.14 without changing the best batch, architecture, or learning-rate schedule.

EVIDENCE: The 176K design performs 2634 steps versus 1833 for the 256K design, while weight decay is applied once per step; scaling 0.20 by 1833/2634 gives 0.139, isolating frequent updates from their unintended 44% increase in decay applications.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.14     # preserve cumulative decay as update count increases
>>>>>>> REPLACE