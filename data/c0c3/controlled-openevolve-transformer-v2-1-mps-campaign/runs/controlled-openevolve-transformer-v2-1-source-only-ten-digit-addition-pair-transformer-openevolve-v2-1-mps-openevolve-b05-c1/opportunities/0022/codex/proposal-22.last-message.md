MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 6 to 5 will lower the model from 1,464 to 1,448 parameters while retaining at least 99% accuracy when trained for 36,000 steps.

INTENDED_EDIT: Use a 5-unit feed-forward layer and extend the default training budget from 32,000 to 36,000 steps.

EVIDENCE: Every successive `d_ff` reduction from 12 through 6 exceeded 99% accuracy when paired with 4,000 additional training steps per removed neuron; most recently, `d_ff=6` achieved 100% accuracy at 32,000 steps with 1,464 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=6)
=======
    p.add_argument("--d-ff", type=int, default=5)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=32000)
=======
    p.add_argument("--train-steps", type=int, default=36000)
>>>>>>> REPLACE