MECHANISM: Higher-frequency optimizer updates

HYPOTHESIS: Halving the optimizer batch to 262K tokens will provide nearly twice as many parameter updates while preserving the successful 50.3M-parameter architecture and final full-context layer, lowering val_bpb below 0.992286 despite modest optimizer overhead.

INTENDED_EDIT: Reduce gradient accumulation from two microbatches to one by halving TOTAL_BATCH_SIZE while retaining DEVICE_BATCH_SIZE and all learning-rate settings.

EVIDENCE: All-local and grouped-query variants processed more tokens but regressed to 1.015479 and 0.996550, showing that token throughput alone is insufficient; increasing optimization cadence tests sample efficiency without sacrificing the empirically essential capacity or global attention.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE