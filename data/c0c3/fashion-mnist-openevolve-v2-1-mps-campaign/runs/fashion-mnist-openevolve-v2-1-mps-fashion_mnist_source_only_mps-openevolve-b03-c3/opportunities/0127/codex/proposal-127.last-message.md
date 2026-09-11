MECHANISM: Reduced classifier dropout under the accuracy-optimal optimizer

HYPOTHESIS: Restoring AdamW beta2=0.96 and reducing classifier dropout from 0.10 to 0.05 will exceed 9,283 correct predictions by improving convergence during the fixed two-pass exposure while retaining augmentation, label smoothing, and weight decay as regularizers.

INTENDED_EDIT: Restore the best verified beta2 and halve only the classifier dropout probability.

EVIDENCE: Beta2=0.96 achieved the best result of 9,283 correct, while nearby optimizer, schedule, augmentation, TTA, and representation changes regressed; dropout strength remains untested despite training already having translation augmentation, paired flips, label smoothing, and weight decay.

<<<<<<< SEARCH
            nn.Dropout(0.10),
=======
            nn.Dropout(0.05),
>>>>>>> REPLACE

<<<<<<< SEARCH
        betas=(0.9, 0.97),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE