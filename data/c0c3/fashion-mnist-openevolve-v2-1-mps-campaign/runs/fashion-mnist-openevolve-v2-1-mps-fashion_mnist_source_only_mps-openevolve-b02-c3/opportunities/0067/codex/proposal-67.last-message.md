MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Changing the EMA decay to 0.98 will reproduce the reference’s 9,344 correct predictions and exceed the current 0.985-decay design’s validation_score.

INTENDED_EDIT: Shorten only the foreach-accelerated full-state EMA decay from 0.985 to 0.98.

EVIDENCE: Reference Design 3’s 0.98 decay achieved the highest observed validation_score, with 9,344 correct and lower cross-entropy than the tied 0.9825 design; the prior restoration attempt was unverifiable and supplied no evidence against the setting.

<<<<<<< SEARCH
                decay = 0.985 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE