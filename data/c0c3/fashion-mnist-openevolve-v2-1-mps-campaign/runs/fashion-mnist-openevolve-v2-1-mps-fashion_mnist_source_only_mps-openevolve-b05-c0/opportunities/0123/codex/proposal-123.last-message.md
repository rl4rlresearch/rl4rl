MECHANISM: Fine-grained late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA coefficient from 0.03 to 0.035 will produce more than 9,260 correct predictions by tracking later training states more closely without discarding the noise suppression that improved prior EMA variants.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.03 to the intermediate value 0.035.

EVIDENCE: Moving the coefficient from 0.01 to 0.02 and then 0.03 added one correct prediction at each step and reduced cross-entropy; because repeated 0.04 attempts timed out without accuracy evidence, 0.035 is the most informative finer-grained continuation of the verified trend.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.035)
>>>>>>> REPLACE