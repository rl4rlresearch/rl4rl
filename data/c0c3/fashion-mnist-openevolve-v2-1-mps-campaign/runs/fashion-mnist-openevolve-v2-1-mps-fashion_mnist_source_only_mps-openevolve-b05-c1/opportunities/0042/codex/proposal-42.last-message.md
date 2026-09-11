MECHANISM: Reduced bottleneck dropout

HYPOTHESIS: Halving classifier dropout from 0.1 to 0.05 will exceed 9,249 correct predictions by reducing optimization noise during the fixed two-pass budget while retaining mild regularization.

INTENDED_EDIT: Reduce only the dropout probability in the existing classifier head; preserve architecture, parameter count, optimizer, augmentation, and calibrated ensemble.

EVIDENCE: Smaller batches and a higher peak learning rate previously improved correctness, whereas architecture changes and tail EMA regressed, motivating a conservative optimization-focused change within the successful model.

<<<<<<< SEARCH
            nn.Dropout(0.1),
=======
            nn.Dropout(0.05),
>>>>>>> REPLACE