MECHANISM: Single-neuron feed-forward width ablation

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower parameters from 1628 to 1611 while retaining at least 99% accuracy, because the failed width-8 model removed four neurons at once and does not establish that the immediately adjacent width 11 is below the capacity threshold.

INTENDED_EDIT: Reduce the single transformer block’s default feed-forward width by one neuron while preserving attention, bias sharing, training, and decoding.

EVIDENCE: The width-12 design achieves 99.76% at 1628 parameters, while width 8 achieves only 69.64%; testing width 11 is the smallest controlled step toward locating the feed-forward capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE