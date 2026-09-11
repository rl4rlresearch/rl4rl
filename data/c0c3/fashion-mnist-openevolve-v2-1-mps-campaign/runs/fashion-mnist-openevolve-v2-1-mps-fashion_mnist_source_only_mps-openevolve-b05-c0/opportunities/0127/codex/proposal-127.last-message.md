MECHANISM: Conservative BatchNorm buffer alignment

HYPOTHESIS: Reducing the halfway-state BatchNorm buffer rewind from 0.25 to 0.20 will retain at least 9,260 correct predictions while lowering validation cross-entropy below 0.2125679.

INTENDED_EDIT: Keep the best verified 0.03 parameter EMA and change only the final floating-point buffer interpolation from 0.25 to 0.20.

EVIDENCE: The 0.25 buffer mixture achieved 9,260 correct, while 0.10 reduced cross-entropy to 0.2120064 but lost seven correct predictions; 0.20 conservatively tests whether part of that cross-entropy gain can be captured without the accuracy loss.

<<<<<<< SEARCH
                    buffer.lerp_(start, 0.25)
=======
                    buffer.lerp_(start, 0.20)
>>>>>>> REPLACE