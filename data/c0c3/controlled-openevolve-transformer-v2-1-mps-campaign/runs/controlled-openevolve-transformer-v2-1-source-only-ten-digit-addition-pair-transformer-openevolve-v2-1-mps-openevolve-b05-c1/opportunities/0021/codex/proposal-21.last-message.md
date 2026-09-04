MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 7 to 6 will lower the model from 1,480 to 1,464 parameters while retaining at least 99% accuracy when trained for 32,000 steps.

INTENDED_EDIT: Use a 6-unit feed-forward layer and extend the default training budget from 28,000 to 32,000 steps.

EVIDENCE: Every successive reduction from `d_ff=12` through `d_ff=7` exceeded 99% accuracy when paired with 4,000 additional steps per removed neuron; most recently, `d_ff=7` achieved 100% accuracy at 28,000 steps.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=7)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=28000)
=======
    p.add_argument("--train-steps", type=int, default=32000)
>>>>>>> REPLACE