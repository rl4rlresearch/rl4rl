MECHANISM: One-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 12 to 11 will lower the model from 1,218 to 1,203 learned parameters while retaining at least 99% accuracy, because the current model reaches 99.99% and the change removes only one of twelve nonlinear hidden units without altering attention or decoding.

INTENDED_EDIT: Change the default feed-forward width to 11; training, optimizer behavior, checkpointing, and the protected generation interface remain unchanged.

EVIDENCE: The 1,218-parameter design achieved 99.99% accuracy after 4,999 steps, providing the strongest available evidence that a conservative capacity reduction can preserve the required 99% threshold.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE