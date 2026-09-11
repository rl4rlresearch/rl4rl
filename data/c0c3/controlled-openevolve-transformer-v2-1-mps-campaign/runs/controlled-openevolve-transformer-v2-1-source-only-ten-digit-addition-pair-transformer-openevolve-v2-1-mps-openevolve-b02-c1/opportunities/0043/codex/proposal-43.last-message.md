MECHANISM: Sixth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 7 to 6 will lower the model from 1,143 to 1,128 learned parameters while retaining at least 99% accuracy, because the width-7 model achieved 100% accuracy after five consecutive successful one-neuron reductions.

INTENDED_EDIT: Change the default feed-forward width from 7 to 6 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 7 each removed 15 parameters while maintaining at least 99.91% accuracy, and the latest width-7 model achieved 100%; this uninterrupted trend makes the next single-neuron ablation the most informative capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=7)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE