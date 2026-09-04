MECHANISM: Short-horizon parameter-only EMA

HYPOTHESIS: Increasing the EMA interpolation rate from 0.01 to 0.02 will exceed 9,282 correct predictions by reducing parameter lag while retaining useful smoothing over roughly the final 50 optimizer steps.

INTENDED_EDIT: Shorten the learned-parameter EMA horizon from approximately 100 steps to 50 steps, leaving the architecture, paired-view objective, BatchNorm buffers, and schedule unchanged.

EVIDENCE: Parameter-only EMA matched the best 9,282 correct while lowering cross-entropy, whereas lengthening the EMA horizon from 0.01 to 0.005 previously reduced correct predictions from 9,237 to 9,233; this directly motivates testing less trajectory lag.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.01)
=======
                        averaged.lerp_(value.detach(), 0.02)
>>>>>>> REPLACE