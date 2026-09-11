MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Restoring the verified 0.98 EMA decay will recover 9,344 correct predictions and exceed the current 0.975-decay design’s validation_score.

INTENDED_EDIT: Change only the foreach-accelerated full-state EMA decay from 0.975 to 0.98.

EVIDENCE: The verified 0.98 design achieved the highest observed validation_score with 9,344 correct and 0.18781964 cross-entropy, while the current 0.975 design regressed to 9,340 correct; prior restoration attempts were unverifiable and provide no contrary performance evidence.

<<<<<<< SEARCH
                decay = 0.975 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE