MECHANISM: Narrower feed-forward bottleneck

HYPOTHESIS: Reducing the sole MLP width from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy.

INTENDED_EDIT: Change the default feed-forward dimension to match the 8-dimensional residual stream.

EVIDENCE: The current 1644-parameter model achieves 99.96% accuracy, providing substantial accuracy margin for a 68-parameter reduction in its MLP.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE