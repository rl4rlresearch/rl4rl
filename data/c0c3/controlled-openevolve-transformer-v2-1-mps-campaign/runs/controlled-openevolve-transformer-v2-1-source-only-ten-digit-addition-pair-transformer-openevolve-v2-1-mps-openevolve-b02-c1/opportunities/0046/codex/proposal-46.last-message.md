MECHANISM: Ninth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 4 to 3 will lower the model from 1,098 to 1,083 learned parameters while retaining at least 99% accuracy, because eight consecutive one-neuron reductions have met the requirement and the width-4 model achieved 99.96%.

INTENDED_EDIT: Change the default feed-forward width from 4 to 3 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 4 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-4 result reached 99.96%, so the next single-neuron ablation is the most direct test of the remaining MLP capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=4)
=======
    p.add_argument("--d-ff", type=int, default=3)
>>>>>>> REPLACE