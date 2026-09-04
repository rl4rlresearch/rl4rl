MECHANISM: Five-unit nonlinear carry bottleneck

HYPOTHESIS: A 962-parameter transformer will maintain at least 99% accuracy because the otherwise identical six-unit, 976-parameter model achieved 100%; reducing the MLP by one further unit tests the same repeatedly successful 14-parameter compression.

INTENDED_EDIT: Adopt the verified six-unit architecture and reduce its learned MLP hidden width to five units while preserving all attention, embedding, projection, normalization, training, checkpointing, and decoding behavior.

EVIDENCE: Successive MLP-width reductions through six units all retained at least 99% accuracy, and the six-unit reference achieved 100%, providing the strongest direct evidence for testing one additional width reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=5)
>>>>>>> REPLACE