MECHANISM: Ten-unit nonlinear carry bottleneck

HYPOTHESIS: A 1,032-parameter transformer will maintain at least 99% accuracy because reducing the MLP from twelve to eleven units removed fourteen parameters while improving the verified result from the 1,060-parameter model’s 99.95% to 100%, indicating remaining nonlinear-width slack.

INTENDED_EDIT: Reduce the learned MLP hidden width from eleven to ten units, removing fourteen projection weights while preserving the verified attention routing, embeddings, biases, residual width, and decoding interface.

EVIDENCE: The current eleven-unit, 1,046-parameter design achieved 100% accuracy after a full fourteen-parameter width reduction; testing the next identical reduction is the clearest measurement of whether that successful bottleneck compression continues.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE