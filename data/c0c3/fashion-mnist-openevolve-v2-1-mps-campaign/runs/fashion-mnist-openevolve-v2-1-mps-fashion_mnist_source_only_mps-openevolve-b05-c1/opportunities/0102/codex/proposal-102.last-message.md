MECHANISM: Short-horizon parameter EMA

HYPOTHESIS: Lowering the EMA decay cap to 0.985 will exceed 9,351 correct predictions by reducing evaluation-state lag while retaining late-training noise suppression.

INTENDED_EDIT: Reduce only the parameter EMA decay cap from 0.99 to 0.985.

EVIDENCE: Keeping live BatchNorm statistics improved validation correct from 9,345 to 9,351, supporting reduced evaluation-state lag; the prior 0.985 EMA attempt timed out despite adding no computation, so its effect remains unresolved.

<<<<<<< SEARCH
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
=======
            ema_decay = min(0.985, (updates + 1.0) / (updates + 10.0))
>>>>>>> REPLACE