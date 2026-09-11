MECHANISM: Eighth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 5 to 4 will lower the model from 1,113 to 1,098 learned parameters while retaining at least 99% accuracy, because seven consecutive one-neuron reductions have met the requirement and the width-5 model achieved 99.92%.

INTENDED_EDIT: Change the default feed-forward width from 5 to 4 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 5 each removed 15 parameters while maintaining at least 99.88% accuracy; the latest width-5 result reached 99.92%, making the next single-neuron ablation the most informative remaining capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=5)
=======
    p.add_argument("--d-ff", type=int, default=4)
>>>>>>> REPLACE