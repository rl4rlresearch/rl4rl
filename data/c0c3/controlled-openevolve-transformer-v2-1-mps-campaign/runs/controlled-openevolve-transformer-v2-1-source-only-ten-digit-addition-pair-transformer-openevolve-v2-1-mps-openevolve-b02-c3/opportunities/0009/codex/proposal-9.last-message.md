MECHANISM: Single-neuron feed-forward capacity reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower parameters from 1583 to 1567 while retaining at least 99% accuracy, because it preserves nearly all nonlinear capacity of the verified 99.90%-accurate design.

INTENDED_EDIT: Change the default feed-forward width from 12 to 11.

EVIDENCE: The current `d_ff=12` quotient-residual design achieved 99.90% accuracy, while the larger reduction to `d_ff=8` failed at 71.49%; testing the smallest possible width reduction is the most informative next capacity-boundary experiment.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE