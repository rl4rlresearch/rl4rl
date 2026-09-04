MECHANISM: Lower-bracket full-state EMA horizon refinement

HYPOTHESIS: A 0.97875-decay full-state EMA will exceed the current validation_score by selecting an averaging horizon between the best 0.98 setting and the regressed 0.975 setting.

INTENDED_EDIT: Change only the foreach-accelerated full-state EMA decay from 0.98 to 0.97875.

EVIDENCE: Decay 0.98 achieved the highest observed score with 9,344 correct, while 0.975 fell to 9,340 and the upper midpoint 0.98125 fell to 9,343; testing the unmeasured lower midpoint is the most focused remaining refinement around the observed optimum.

<<<<<<< SEARCH
                decay = 0.98 ** elapsed
=======
                decay = 0.97875 ** elapsed
>>>>>>> REPLACE