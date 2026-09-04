MECHANISM: Dropout-free narrow classifier head

HYPOTHESIS: Removing 0.1 dropout from the 38-unit classifier head will increase validation_correct above 9,247 by improving convergence within the fixed two-pass exposure budget.

INTENDED_EDIT: Disable the classifier-head dropout while preserving the verified architecture, EMA, augmentation, optimizer schedule, and calibrated flip ensemble.

EVIDENCE: Adding 0.05 label smoothing reduced validation_correct from 9,243 to 9,225 and worsened cross-entropy, indicating that additional regularization impedes this short training regime; the verified EMA already supplies parameter averaging and regularization.

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Dropout(p=0.0),
>>>>>>> REPLACE