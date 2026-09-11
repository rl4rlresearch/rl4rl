MECHANISM: Upper-side BatchNorm-buffer EMA refinement

HYPOTHESIS: A 3.25% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly improving responsiveness while preserving the smoothing benefit of the best-tested 3% rate.

INTENDED_EDIT: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.25%.

EVIDENCE: The 3% buffer update achieved the best result at 9,359 correct; 2.5% tied but had worse cross-entropy, while the larger 3.5% test timed out, making 3.25% the closest untested upper-side refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(buffer.detach(), 0.0325)
>>>>>>> REPLACE