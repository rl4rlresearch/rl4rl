MECHANISM: Single-unit MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will retain at least 99% accuracy while reducing learned parameters from 1,644 to 1,627.

INTENDED_EDIT: Preserve the full-width token and positional embeddings and reduce the transformer block’s feed-forward hidden width by one unit.

EVIDENCE: The full-width `d_model=8, d_ff=12` model achieved 99.96% accuracy, while rank-6 embedding factorization collapsed to 7.69%; this motivates preserving embedding expressivity and testing a small reduction in a different subsystem.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE