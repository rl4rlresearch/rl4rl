MECHANISM: Dense-bottleneck batch normalization

HYPOTHESIS: Normalizing the 30-feature classifier bottleneck will exceed 9,249 correct predictions by stabilizing its optimization within the fixed two-pass budget while preserving the successful representation and regularization level.

INTENDED_EDIT: Insert BatchNorm1d after the first classifier linear layer, adding only 60 learned parameters.

EVIDENCE: Changing dropout in either direction reduced correctness, and larger architectural changes also regressed; this tests deterministic bottleneck conditioning without altering dropout, augmentation, schedule, or inference fusion.

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 30),
            nn.GELU(),
=======
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
>>>>>>> REPLACE