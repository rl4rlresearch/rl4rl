MECHANISM: Incrementally shorter parameter EMA horizon

HYPOTHESIS: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by further reducing evaluation-weight lag while preserving smoothing over roughly 40 late-training updates.

INTENDED_EDIT: Reduce only the parameter EMA decay cap from 0.98 to 0.975.

EVIDENCE: Live BatchNorm buffers improved validation correct from 9,345 to 9,351, then EMA caps of 0.985 and 0.98 improved it to 9,357 and 9,358 respectively; the prior 0.975 attempt timed out without producing contrary accuracy evidence.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.975, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE