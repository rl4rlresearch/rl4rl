MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Changing the EMA decay from 0.985 to the verified optimum of 0.98 will recover 9,344 correct predictions and exceed the current 9,343-correct validation score.

INTENDED_EDIT: Shorten only the foreach-accelerated full-state EMA decay from 0.985 to 0.98.

EVIDENCE: Reference Design 3 achieved the highest observed validation score with decay 0.98: 9,344 correct and 0.18781964 cross-entropy. Both adjacent refinements, 0.97875 and 0.98125, lost correct predictions, while prior restoration failures supplied no contradictory performance result.

<<<<<<< SEARCH
                decay = 0.985 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE