MECHANISM: Dense-representation batch normalization

HYPOTHESIS: Normalizing the 128-unit classifier representation will exceed 9,281 correct predictions by improving optimization of the successful feature extractor while remaining under the 250,000-parameter ceiling.

INTENDED_EDIT: Add BatchNorm1d between the classifier’s first linear layer and GELU; retain the verified architecture, loss, schedule, dropout, and evaluation calibration.

EVIDENCE: Spatial channel-mixing, gating, depthwise refinement, augmentation, ensembling, and weight averaging all underperformed the 9,281-correct baseline; this tests a distinct, lightweight classifier-conditioning change while preserving the best feature extractor and adding only 256 learned parameters.

<<<<<<< SEARCH
            nn.Linear(128 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(p=0.05),
=======
            nn.Linear(128 * 3 * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.05),
>>>>>>> REPLACE