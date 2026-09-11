MECHANISM: Update-count-normalized cautious weight decay

HYPOTHESIS: Halving Muon weight decay to preserve its cumulative horizon after the batch-size-driven doubling of optimizer steps will reduce val_bpb below 0.987174 without affecting throughput.

INTENDED_EDIT: Reduce cautious Muon weight decay from 0.2 to 0.1 while retaining the winning architecture, batch size, learning rates, and schedules.

EVIDENCE: Halving the batch doubled update frequency and improved val_bpb, but left per-step weight decay unchanged; the uniform learning-rate reduction nearly tied the best result while also weakening decay, motivating isolation of weight decay without reducing productive parameter updates.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.1      # preserve cumulative decay horizon after doubling optimizer updates
>>>>>>> REPLACE