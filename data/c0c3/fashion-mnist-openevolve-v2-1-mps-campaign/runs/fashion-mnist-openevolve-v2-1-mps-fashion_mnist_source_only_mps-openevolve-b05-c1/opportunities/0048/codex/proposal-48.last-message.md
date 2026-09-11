MECHANISM: Smoothed bottleneck running-statistics estimation

HYPOTHESIS: Reducing the successful bottleneck BatchNorm momentum from 0.1 to 0.05 will exceed 9,314 correct predictions by producing less noisy evaluation statistics from batch-size-96 training while retaining its optimization benefit.

INTENDED_EDIT: Set the 30-feature BatchNorm1d momentum to 0.05; preserve all architecture, optimization, augmentation, and inference behavior.

EVIDENCE: Adding bottleneck BatchNorm improved correctness from 9,249 to 9,314, whereas adding trunk BatchNorm reduced it to 9,305; this motivates refining the proven bottleneck normalization rather than adding normalization elsewhere.

<<<<<<< SEARCH
            nn.BatchNorm1d(30),
=======
            nn.BatchNorm1d(30, momentum=0.05),
>>>>>>> REPLACE