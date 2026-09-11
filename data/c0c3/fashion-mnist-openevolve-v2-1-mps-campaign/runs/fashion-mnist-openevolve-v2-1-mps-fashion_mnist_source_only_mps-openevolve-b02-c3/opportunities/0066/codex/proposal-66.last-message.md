MECHANISM: Best-observed full-state EMA horizon

HYPOTHESIS: Restoring the empirically superior 0.98 EMA decay will retain 9,344 correct predictions while lowering cross-entropy below the current 0.18783146.

INTENDED_EDIT: Change only the full-state EMA decay from 0.9825 to 0.98.

EVIDENCE: The 0.98 reference achieved the best observed validation_score, matching the current design’s 9,344 correct predictions with lower cross-entropy (0.18781964 versus 0.18783146); the intermediate 0.98125 test also lost one correct prediction.

<<<<<<< SEARCH
                decay = 0.9825 ** elapsed
=======
                decay = 0.98 ** elapsed
>>>>>>> REPLACE