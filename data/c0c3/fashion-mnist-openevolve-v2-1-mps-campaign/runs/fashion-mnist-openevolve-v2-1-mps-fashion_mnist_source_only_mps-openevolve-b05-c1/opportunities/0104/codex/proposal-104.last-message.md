MECHANISM: Incrementally shorter parameter EMA horizon

HYPOTHESIS: Lowering the EMA decay cap from 0.98 to 0.975 will exceed 9,358 correct predictions by further reducing evaluation-weight lag while retaining smoothing over roughly 40 late-training updates.

INTENDED_EDIT: Reduce only the parameter EMA decay cap from 0.98 to 0.975.

EVIDENCE: Keeping BatchNorm buffers live improved validation correct from 9,345 to 9,351, then lowering parameter EMA from 0.99 to 0.985 improved it to 9,357 and lowering it again to 0.98 improved it to 9,358 with better cross-entropy; a small additional reduction directly tests whether that consistent trend continues.

<<<<<<< SEARCH
            ema_decay = min(0.98, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.975, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE