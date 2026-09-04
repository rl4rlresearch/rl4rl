MECHANISM: Update-count-normalized cautious weight decay

HYPOTHESIS: On the efficient 128-sequence, 262K-token update design, halving cautious weight decay will beat val_bpb 0.98713 while retaining roughly 490M-token throughput, because twice as many optimizer steps otherwise apply approximately twice as many decay opportunities.

INTENDED_EDIT: Restore the best verified single-microbatch configuration and reduce Muon weight decay from 0.2 to 0.1.

EVIDENCE: Reference Design 3 achieved the best val_bpb, 0.98713, with 1,868 steps and 489.7M tokens; compared with the 948-step baseline, its unchanged per-step decay substantially increases cumulative regularization.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.1      # normalize cumulative cautious decay for the higher update count
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # two microbatches per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE