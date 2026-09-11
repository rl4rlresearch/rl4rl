MECHANISM: Tenth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 3 to 2 will lower the model from 1,083 to 1,068 learned parameters while retaining at least 99% accuracy, because nine consecutive one-neuron reductions have met the requirement and the width-3 model achieved 99.94%.

INTENDED_EDIT: Change the default feed-forward width from 3 to 2 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 3 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-3 result reached 99.94%, making the next single-neuron ablation the most informative remaining capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=3)
=======
    p.add_argument("--d-ff", type=int, default=2)
>>>>>>> REPLACE