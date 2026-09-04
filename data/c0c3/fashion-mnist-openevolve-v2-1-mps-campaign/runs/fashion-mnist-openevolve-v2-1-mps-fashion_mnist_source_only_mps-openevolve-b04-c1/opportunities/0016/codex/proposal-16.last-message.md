MECHANISM: Very-short-horizon parameter-only EMA

HYPOTHESIS: Increasing the EMA interpolation rate from 0.02 to 0.04 will exceed 9,286 correct predictions by further reducing end-of-training parameter lag while retaining smoothing over roughly 25 optimizer steps.

INTENDED_EDIT: Shorten the learned-parameter EMA horizon from approximately 50 steps to 25 steps, leaving the proven architecture, paired-view objective, BatchNorm handling, and learning-rate schedule unchanged.

EVIDENCE: Raising the parameter-only EMA rate from 0.01 to 0.02 improved validation correct from 9,282 to 9,286, while the earlier longer-horizon 0.005 EMA regressed; another geometric reduction directly tests the observed benefit of lower EMA lag.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.04)
>>>>>>> REPLACE