MECHANISM: Stronger fixed dense-head dropout

HYPOTHESIS: Increasing dense-head dropout from 10% to 15% will exceed 9,290 correct predictions by strengthening the regularization whose removal reduced accuracy, without the runtime risk of a per-step dropout curriculum.

INTENDED_EDIT: Raise the classifier dropout probability from 0.10 to 0.15 while preserving architecture, optimization, augmentation, and ensembling.

EVIDENCE: Removing 10% dropout reduced validation correct from 9,290 to 9,270, demonstrating that dense-head dropout is beneficial; the scheduled-dropout experiment timed out, motivating a static increase as the cleanest test of stronger regularization.

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Dropout(p=0.15),
>>>>>>> REPLACE