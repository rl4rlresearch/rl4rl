MECHANISM: Update-count-normalized cautious decay

HYPOTHESIS: Restoring the efficient 262K-token batch while halving Muon weight decay will lower val_bpb below 0.987182 by correcting the doubled per-token decay caused by twice as many optimizer steps.

INTENDED_EDIT: Restore the proven 128-sequence single microbatch and reduce cautious Muon weight decay from 0.2 to 0.1 without changing momentum or variance averaging.

EVIDENCE: The 262K-token design achieved the best val_bpb, 0.987182, with 1,868 updates versus 948 for the 524K baseline; momentum-ramp and beta2 changes both worsened it, motivating retention of its optimizer statistics while isolating accumulated per-step decay.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step; one device batch
=======
TOTAL_BATCH_SIZE = 2**18     # ~262K tokens per optimizer step; one device batch
>>>>>>> REPLACE

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.1      # normalize cumulative cautious decay for the higher update cadence
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE