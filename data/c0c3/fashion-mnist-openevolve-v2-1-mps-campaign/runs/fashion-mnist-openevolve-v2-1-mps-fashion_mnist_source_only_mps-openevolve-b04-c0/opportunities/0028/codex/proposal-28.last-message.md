MECHANISM: Lower-side BatchNorm-buffer EMA refinement

HYPOTHESIS: A 2.75% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by interpolating between the accuracy-preserving 2.5% rate and the best-tested 3% rate while keeping parameter EMA unchanged.

INTENDED_EDIT: Keep parameter EMA at 4% and reduce only the floating-buffer EMA update from 3% to 2.75%.

EVIDENCE: The 3% buffer update achieved 9,359 correct with 0.193865 cross-entropy, while 2.5% also achieved 9,359 correct but with worse cross-entropy; their untested midpoint is the closest remaining local refinement.

<<<<<<< SEARCH
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(buffer.detach(), 0.0275)
>>>>>>> REPLACE