MECHANISM: Stateless dense-head layer normalization

HYPOTHESIS: Per-example normalization of the parameter-dominant 160-unit classifier representation will exceed 9,166 correct predictions by stabilizing short-budget optimization without introducing BatchNorm running-state mismatch or learned parameters.

INTENDED_EDIT: Insert non-affine LayerNorm between the first classifier linear layer and GELU, retaining 249,789 learned parameters and all other training behavior.

EVIDENCE: Zero-DC adaptive detail filters established the 9,166-correct baseline; the related non-affine BatchNorm head experiment timed out, so a stateless, batch-independent normalization is the most direct unresolved test of dense-head stabilization.

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
=======
            nn.Linear(96 * 3 * 3, 160),
            nn.LayerNorm(160, elementwise_affine=False),
            nn.GELU(),
>>>>>>> REPLACE