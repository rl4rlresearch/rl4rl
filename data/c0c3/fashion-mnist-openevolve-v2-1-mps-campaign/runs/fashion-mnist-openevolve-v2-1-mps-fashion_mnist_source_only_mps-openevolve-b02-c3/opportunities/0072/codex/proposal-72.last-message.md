MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Restoring decay 0.98 will recover the verified 9,344 correct predictions and exceed the current 0.975-decay design’s validation score.

INTENDED_EDIT: Change only the foreach-accelerated full-state EMA decay from 0.975 to 0.98.

EVIDENCE: The verified 0.98 design achieved the highest observed score with 9,344 correct and 0.18781964 cross-entropy; 0.975 regressed to 9,340, while unsuccessful restoration attempts produced no contradictory validation result.

<<<<<<< SEARCH
                decay = 0.975 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE