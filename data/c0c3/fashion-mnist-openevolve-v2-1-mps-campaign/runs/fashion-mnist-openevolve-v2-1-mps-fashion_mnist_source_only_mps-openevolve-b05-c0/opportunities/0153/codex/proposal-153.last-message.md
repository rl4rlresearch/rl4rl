MECHANISM: Terminal-weight/EMA interpolation

HYPOTHESIS: Retaining 10% of the final optimizer iterate when installing the EMA weights will exceed 9,260 correct predictions by preserving late decision-boundary refinement while retaining most EMA variance reduction.

INTENDED_EDIT: Replace the pure EMA parameter copy with a 90% EMA, 10% terminal-weight interpolation; preserve architecture, optimizer, loss, schedule, BatchNorm mixture, and calibration.

EVIDENCE: The immediate-cosine baseline remains strongest, while warmup improved cross-entropy but reduced correctness to 9,250 and both learning-rate increases also reduced correctness or timed out; this motivates a conservative change to late-weight averaging without perturbing the verified training trajectory.

<<<<<<< SEARCH
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
=======
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.lerp_(average, 0.9)
>>>>>>> REPLACE