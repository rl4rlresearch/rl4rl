MECHANISM: Seventh one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 6 to 5 will lower the model from 1,128 to 1,113 learned parameters while retaining at least 99% accuracy, because six consecutive one-neuron reductions have preserved the requirement and the width-6 model achieved 99.88%.

INTENDED_EDIT: Change the default feed-forward width from 6 to 5 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 6 each removed 15 parameters while maintaining at least 99.88% accuracy; continuing this demonstrated trend is the most direct test of the remaining MLP capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=6)
=======
    p.add_argument("--d-ff", type=int, default=5)
>>>>>>> REPLACE