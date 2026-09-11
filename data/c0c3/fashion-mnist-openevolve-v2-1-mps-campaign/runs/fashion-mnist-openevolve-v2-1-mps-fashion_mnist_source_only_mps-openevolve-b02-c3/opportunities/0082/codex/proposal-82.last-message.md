MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Restoring EMA decay 0.98 will retain 9,344 correct predictions while lowering validation cross-entropy below the current 0.18783146.

INTENDED_EDIT: Change only the full-state EMA decay from 0.9825 to 0.98.

EVIDENCE: Reference Design 3 used decay 0.98 and achieved 9,344 correct with lower cross-entropy of 0.18781964, strictly outperforming the current equal-correct design.

<<<<<<< SEARCH
                decay = 0.9825 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE