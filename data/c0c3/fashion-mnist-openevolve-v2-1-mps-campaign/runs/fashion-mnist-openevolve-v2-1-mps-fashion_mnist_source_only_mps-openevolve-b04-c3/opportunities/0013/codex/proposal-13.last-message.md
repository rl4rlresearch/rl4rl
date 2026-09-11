MECHANISM: Stronger bottleneck dropout

HYPOTHESIS: Increasing classifier dropout from 0.10 to 0.15 in the qualified batch-64 model will exceed 9,229 correct predictions by modestly strengthening the head regularization that already outperformed no dropout.

INTENDED_EDIT: Restore the qualified batch size of 64 and increase the existing classifier dropout probability to 0.15 without changing parameters or computational structure.

EVIDENCE: Reference Design 2 achieved 9,229 correct with 10% dropout, while removing dropout reduced correctness to 9,225 despite slightly lower cross-entropy; this directly motivates testing a small increase in dropout to prioritize the accuracy-ranked objective.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Dropout(p=0.15),
>>>>>>> REPLACE