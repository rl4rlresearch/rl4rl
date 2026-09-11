MECHANISM: Width-matched feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower deduplicated parameters from 1,644 to 1,576 while retaining at least 99% accuracy because attention width, embeddings, and training budget remain unchanged.

INTENDED_EDIT: Set the single transformer block’s feed-forward width equal to its model width.

EVIDENCE: The current `d_ff=12` design reached 99.96% accuracy with 1,644 parameters, providing margin for a targeted 68-parameter reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE