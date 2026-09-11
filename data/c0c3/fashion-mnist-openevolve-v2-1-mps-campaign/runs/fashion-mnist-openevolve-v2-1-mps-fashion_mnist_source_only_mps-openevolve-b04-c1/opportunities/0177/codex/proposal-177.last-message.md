MECHANISM: EMA-aligned BatchNorm statistics

HYPOTHESIS: Averaging BatchNorm running means and variances at the backbone’s 0.02 EMA rate will exceed 9,330 correct predictions by aligning normalization statistics with the averaged feature weights.

INTENDED_EDIT: EMA floating-point BatchNorm buffers during second-half averaging while continuing to copy integer counters directly.

EVIDENCE: Classifier-specific parameter EMA produced the best 9,330-result, while architecture and translation changes underperformed or timed out; all reported EMA experiments retained final-step BatchNorm buffers, leaving weight/statistic alignment untested.

<<<<<<< SEARCH
                    else:
                        averaged.copy_(value)
=======
                    elif (
                        name.endswith("running_mean")
                        or name.endswith("running_var")
                    ):
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE