MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 11 to 10 will lower the model from 1,544 to 1,528 parameters while retaining at least 99% accuracy when trained for 16,000 steps.

INTENDED_EDIT: Use a 10-unit feed-forward layer and extend the default training budget from 12,000 to 16,000 steps.

EVIDENCE: The quotient-space model reached 99.87% with `d_ff=12` at 8,000 steps, and `d_ff=11` reached 99.93% at 12,000 steps; this supports testing the next width boundary with another 4,000-step convergence allowance.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=12000)
=======
    p.add_argument("--train-steps", type=int, default=16000)
>>>>>>> REPLACE