MECHANISM: Shorter-horizon full-state EMA

HYPOTHESIS: Reducing EMA decay from 0.98 to 0.9775 will exceed 9,344 correct predictions by tracking the final low-learning-rate solution more closely while retaining full-state averaging.

INTENDED_EDIT: Shorten only the full-state EMA horizon, preserving the best architecture, training procedure, and center-weight-3 arithmetic TTA.

EVIDENCE: Lowering decay from 0.9825 to 0.98 retained 9,344 correct while improving cross-entropy from 0.18783146 to 0.18781964; an equal-sized decrement directly tests whether the favorable shorter-horizon trend continues.

<<<<<<< SEARCH
                decay = 0.98 ** elapsed
=======
                decay = 0.9775 ** elapsed
>>>>>>> REPLACE