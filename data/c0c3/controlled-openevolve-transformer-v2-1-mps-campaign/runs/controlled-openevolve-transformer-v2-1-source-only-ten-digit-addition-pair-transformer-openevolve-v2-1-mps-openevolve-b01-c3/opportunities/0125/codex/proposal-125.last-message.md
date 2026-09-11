MECHANISM: Six-unit nonlinear carry bottleneck

HYPOTHESIS: A 976-parameter transformer will maintain at least 99% accuracy because five successive MLP-width reductions through seven units retained at least 99%, and the verified seven-unit 990-parameter model achieved 100%; reducing the width by one tests the same isolated 14-parameter reduction.

INTENDED_EDIT: Reduce the MLP hidden width from seven to six units while preserving the verified attention, embeddings, projections, biases, residual width, training procedure, and decoding interface.

EVIDENCE: The otherwise corresponding eight-unit and seven-unit models achieved 99.99% and 100% accuracy at 1,004 and 990 parameters, respectively, directly motivating one further bottleneck-width reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=7)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE