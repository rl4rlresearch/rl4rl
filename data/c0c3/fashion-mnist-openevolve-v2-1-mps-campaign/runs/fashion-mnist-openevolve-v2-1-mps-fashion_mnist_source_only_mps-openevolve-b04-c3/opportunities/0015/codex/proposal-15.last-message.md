MECHANISM: Light bottleneck dropout

HYPOTHESIS: Reducing classifier dropout from 0.10 to 0.05 will exceed 9,229 correct predictions by retaining useful head regularization while reducing stochastic feature loss during the fixed two-pass training budget.

INTENDED_EDIT: Change only the classifier dropout probability from 0.10 to 0.05, preserving the qualified architecture, batch size, augmentation, optimizer, schedule, and logit-space flip ensemble.

EVIDENCE: The qualified 0.10-dropout model achieved 9,229 correct, no dropout remained close at 9,225 with slightly better cross-entropy, and stronger 0.15 dropout fell to 9,190; this motivates testing lighter—not stronger—regularization.

<<<<<<< SEARCH
            nn.Dropout(p=0.1),
=======
            nn.Dropout(p=0.05),
>>>>>>> REPLACE