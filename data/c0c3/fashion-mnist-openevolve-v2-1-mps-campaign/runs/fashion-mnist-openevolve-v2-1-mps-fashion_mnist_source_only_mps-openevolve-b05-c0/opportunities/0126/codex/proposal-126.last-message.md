MECHANISM: Late-state BatchNorm alignment

HYPOTHESIS: Reducing the halfway-state BatchNorm buffer rewind from 25% to 10% will exceed 9,260 correct predictions by aligning normalization statistics more closely with the late-state EMA weights.

INTENDED_EDIT: Preserve the best verified 0.03 parameter EMA and reduce only the final BatchNorm-buffer interpolation toward halfway-training statistics.

EVIDENCE: The 0.03 EMA achieved the best verified result of 9,260 correct; increasing it to 0.035 fell to 9,257, so independently refining buffer alignment is more informative. The prior 0.10 buffer attempt timed out and supplied no contradictory validation evidence.

<<<<<<< SEARCH
                    buffer.lerp_(start, 0.25)
=======
                    buffer.lerp_(start, 0.10)
>>>>>>> REPLACE