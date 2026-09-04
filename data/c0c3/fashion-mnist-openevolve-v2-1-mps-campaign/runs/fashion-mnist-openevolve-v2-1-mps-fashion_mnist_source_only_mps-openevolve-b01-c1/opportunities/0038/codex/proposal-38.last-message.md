MECHANISM: Smaller-batch trajectory refinement

HYPOTHESIS: Reducing the batch size from 128 to 64 while halving the learning-rate schedule will exceed 9,239 correct predictions by doubling optimizer updates and increasing gradient diversity without changing the example-normalized optimization magnitude.

INTENDED_EDIT: Train with batch size 64 and scale both the initial and scheduled learning rates by 0.5; preserve the architecture, six-view objective, cosine supervision curriculum, and AdamW configuration.

EVIDENCE: The cosine curriculum reached 9,239 correct, while further curriculum shaping and a residual refinement block regressed to 9,236 and 9,188; changing exposure granularity tests an orthogonal optimization mechanism while retaining the strongest established design.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
        weight_decay=2.0e-4,
=======
        lr=1.5e-4,
        weight_decay=2.0e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 1.5e-3 * multiplier
>>>>>>> REPLACE