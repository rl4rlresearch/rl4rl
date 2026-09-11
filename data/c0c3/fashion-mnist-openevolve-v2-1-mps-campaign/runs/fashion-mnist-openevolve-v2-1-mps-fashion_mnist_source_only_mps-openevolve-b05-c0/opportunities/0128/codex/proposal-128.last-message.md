MECHANISM: Conservative BatchNorm buffer interpolation

HYPOTHESIS: Reducing the halfway-state BatchNorm buffer rewind from 0.25 to 0.225 will retain at least 9,260 correct predictions while lowering validation cross-entropy below 0.2125679.

INTENDED_EDIT: Preserve the best verified 0.03 parameter EMA and change only the final floating-point buffer interpolation coefficient from 0.25 to 0.225.

EVIDENCE: The 0.25 mixture achieved 9,260 correct, while 0.10 lowered cross-entropy to 0.2120064 but lost seven correct predictions; 0.225 is a conservative step toward the cross-entropy-improving direction, and the uncompleted 0.20 attempt provides no contradictory accuracy evidence.

<<<<<<< SEARCH
                    buffer.lerp_(start, 0.25)
=======
                    buffer.lerp_(start, 0.225)
>>>>>>> REPLACE