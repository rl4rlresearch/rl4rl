MECHANISM: Short-horizon parameter EMA

HYPOTHESIS: Lowering the EMA decay cap to 0.985 will exceed 9,351 correct predictions by reducing parameter lag while retaining late-training noise suppression.

INTENDED_EDIT: Reduce the parameter EMA decay cap from 0.99 to 0.985; architecture, BatchNorm buffers, optimizer, schedule, and runtime remain unchanged.

EVIDENCE: Keeping live BatchNorm statistics improved the best result from 9,345 to 9,351 correct, showing that reducing evaluation-state lag is beneficial; shortening the parameter EMA horizon tests the same mechanism directly.

<<<<<<< SEARCH
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.985, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE