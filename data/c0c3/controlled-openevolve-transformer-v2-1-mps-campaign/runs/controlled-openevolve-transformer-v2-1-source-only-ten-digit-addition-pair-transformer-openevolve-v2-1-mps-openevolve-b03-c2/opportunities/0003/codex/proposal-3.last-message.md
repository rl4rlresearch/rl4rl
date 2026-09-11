MECHANISM: Single-neuron feed-forward boundary search

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will retain at least 99% accuracy while reducing parameters from 1644 to 1627.

INTENDED_EDIT: Change the default feed-forward width from 12 to 11.

EVIDENCE: Width 12 achieved 99.96% accuracy, while widths 10 and 8 failed; width 11 is the remaining untested point at the observed capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE