MECHANISM: Five-unit nonlinear bottleneck

HYPOTHESIS: Reducing the current six-unit MLP to five units will retain at least 99% accuracy while lowering the learned parameter count from 748 to 734.

INTENDED_EDIT: Change the default feed-forward width from six to five, isolating a 14-parameter reduction without altering attention, embeddings, training, or decoding.

EVIDENCE: The current 748-parameter rank-five model achieved 99.88% accuracy with a six-unit MLP, while the references show repeated MLP-width reductions retained at least 99%; this makes the next one-unit width reduction the most direct test of remaining nonlinear-capacity redundancy.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=6)
=======
    p.add_argument("--d-ff", type=int, default=5)
>>>>>>> REPLACE