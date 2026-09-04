MECHANISM: Narrow residual stream with preserved two-head attention

HYPOTHESIS: Reducing `d_model` from 8 to 6 will lower deduplicated parameters from 1644 to 1188 while retaining at least 99% accuracy because the verified 99.96% baseline has substantial accuracy margin and the two attention heads plus 12-wide MLP remain intact.

INTENDED_EDIT: Change the default residual width to 6 while preserving the existing architecture, head count, training schedule, and decoding interface.

EVIDENCE: The current 8-wide model achieves 0.9996 accuracy with 1644 parameters after 4999 steps, motivating a direct width reduction to locate the capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
=======
    p.add_argument("--d-model", type=int, default=6)
    p.add_argument("--n-head", type=int, default=2)
>>>>>>> REPLACE