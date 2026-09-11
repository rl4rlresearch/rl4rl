MECHANISM: Parameter-free dense-head batch normalization

HYPOTHESIS: Normalizing the 160-unit hidden classifier representation will exceed 9,162 correct predictions by stabilizing optimization of the parameter-dominant dense head without consuming the remaining parameter budget.

INTENDED_EDIT: Insert non-affine BatchNorm1d between the first classifier linear layer and GELU, preserving the 249,762 learned-parameter count.

EVIDENCE: Dual-statistic attention reached 9,162 correct, while subsequent attention refinements and the added edge-strength input failed to improve; this motivates an orthogonal optimization change to the currently unnormalized classifier head.

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
=======
            nn.Linear(96 * 3 * 3, 160),
            nn.BatchNorm1d(160, affine=False),
            nn.GELU(),
            nn.Dropout(0.10),
>>>>>>> REPLACE