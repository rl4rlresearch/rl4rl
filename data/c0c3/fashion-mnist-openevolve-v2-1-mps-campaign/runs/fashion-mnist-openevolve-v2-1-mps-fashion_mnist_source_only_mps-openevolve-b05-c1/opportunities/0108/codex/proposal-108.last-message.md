MECHANISM: Incrementally shorter parameter EMA horizon

HYPOTHESIS: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by continuing the verified benefit of reducing evaluation-weight lag while retaining smoothing over roughly 40 updates.

INTENDED_EDIT: Reduce only the learned-parameter EMA decay cap from 0.98 to 0.975.

EVIDENCE: Keeping BatchNorm buffers live improved validation correct from 9,345 to 9,351, and lowering the parameter EMA cap from 0.99 to 0.985 and then 0.98 improved it to 9,357 and 9,358; prior 0.975 attempts timed out despite identical computational cost and therefore provide no contrary accuracy evidence.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.975, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE