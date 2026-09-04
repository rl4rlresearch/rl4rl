MECHANISM: Further-shortened parameter EMA horizon

HYPOTHESIS: Lowering the EMA decay cap from 0.985 to 0.98 will exceed 9,357 correct predictions by further reducing evaluation-weight lag while preserving smoothing over roughly 50 late-training updates.

INTENDED_EDIT: Reduce only the parameter EMA decay cap from 0.985 to 0.98.

EVIDENCE: Removing EMA from BatchNorm buffers improved validation correct from 9,345 to 9,351, and lowering the parameter EMA cap from 0.99 to 0.985 further improved it to 9,357; this consistent direction motivates testing another measured reduction in parameter lag.

<<<<<<< SEARCH
            ema_decay = min(0.985, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE