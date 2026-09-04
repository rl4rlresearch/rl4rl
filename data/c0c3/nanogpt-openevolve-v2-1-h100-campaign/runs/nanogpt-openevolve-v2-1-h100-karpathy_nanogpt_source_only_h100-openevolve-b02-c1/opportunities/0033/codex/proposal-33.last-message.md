MECHANISM: Batch-transfer-normalized cautious Muon decay

HYPOTHESIS: A 192K-token update with weight decay scaled to preserve the best 176K regime’s cumulative decay will reduce gradient noise and lower val_bpb below 0.984068.

INTENDED_EDIT: Increase the single-microbatch update from 88 to 96 sequences and scale Muon weight decay from 0.126 to 0.138.

EVIDENCE: At weight decay 0.20, the 176K and 192K batches were essentially tied at 0.986967 and 0.986971, while update-count-normalized decay subsequently improved the 176K design to 0.984068; scaling 0.126 by 2655/2429 transfers that successful decay exposure to the less noisy 192K batch.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 11 * 2**14 # 176K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 12 * 2**14 # 192K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
WEIGHT_DECAY = 0.126    # interpolate the bracketed cautious-decay optimum
=======
WEIGHT_DECAY = 0.138    # preserve cumulative decay with fewer 192K-batch updates
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 88   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE