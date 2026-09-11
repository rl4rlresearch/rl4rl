MECHANISM: Late-state BatchNorm alignment

HYPOTHESIS: Reducing the halfway-state BatchNorm buffer rewind from 25% to 10% will increase validation_correct above 9,260 by better aligning normalization statistics with the late-state EMA weights.

INTENDED_EDIT: Keep the best verified 0.03 parameter EMA and reduce only the final interpolation toward halfway-training floating-point buffers.

EVIDENCE: EMA coefficients 0.01, 0.02, and 0.03 successively improved validation_correct by favoring later parameter states, while 0.035 reduced accuracy; this motivates preserving 0.03 and independently shifting the fixed BatchNorm-buffer mixture toward later statistics.

<<<<<<< SEARCH
                    buffer.lerp_(start, 0.25)
=======
                    buffer.lerp_(start, 0.10)
>>>>>>> REPLACE