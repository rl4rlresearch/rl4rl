MECHANISM: Eleven-unit nonlinear carry bottleneck

HYPOTHESIS: A 1,046-parameter transformer will maintain at least 99% accuracy because the verified 1,060-parameter model reached 99.95%, and reducing the MLP from twelve to eleven learned nonlinear features preserves all attention routing, seven-dimensional residual features, biases, and embeddings while testing whether the previously fixed hidden width is overparameterized.

INTENDED_EDIT: Replace the twelve-unit MLP with an eleven-unit MLP, removing fourteen learned projection weights and challenging the prior assumption that further compression must come from tying attention-output or secondary-head coordinates.

EVIDENCE: The current model has a 0.95-point accuracy margin, while gain compression to three feature groups still achieved 99.23%; this suggests the learned computation tolerates representational compression, making removal of one independently learned nonlinear MLP feature a plausible and substantially larger reduction than another scalar tie.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE