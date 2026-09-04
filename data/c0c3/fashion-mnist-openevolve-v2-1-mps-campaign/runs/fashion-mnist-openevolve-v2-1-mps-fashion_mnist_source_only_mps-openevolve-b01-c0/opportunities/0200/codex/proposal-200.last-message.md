MECHANISM: Longer-horizon late-training EMA

HYPOTHESIS: Increasing EMA decay from 0.99 to 0.995 will reduce checkpoint noise and raise validation correctness above 9,290 by averaging roughly 200 rather than 100 late optimizer updates.

INTENDED_EDIT: Change only the EMA decay while preserving architecture, training exposure, inference blending, and the best verified temperature.

EVIDENCE: Float32 temperature searches around 0.717663049697876 preserved the same 9,290 predictions and failed to improve its cross-entropy, so the next informative test must alter predictions; the existing live/EMA ensemble provides a low-risk generalization axis.

<<<<<<< SEARCH
            decay = 0.99
=======
            decay = 0.995
>>>>>>> REPLACE