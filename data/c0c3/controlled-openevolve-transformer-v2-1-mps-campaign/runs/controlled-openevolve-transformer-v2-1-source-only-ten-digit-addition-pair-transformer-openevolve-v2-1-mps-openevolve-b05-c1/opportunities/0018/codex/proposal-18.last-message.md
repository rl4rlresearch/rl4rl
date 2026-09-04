MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 9 to 8 will lower the model from 1,512 to 1,496 parameters while retaining at least 99% accuracy when trained for 24,000 steps.

INTENDED_EDIT: Use an 8-unit feed-forward layer and extend the default training budget from 20,000 to 24,000 steps.

EVIDENCE: The quotient-space model remained successful through successive `d_ff` reductions from 12 to 9 with 4,000 additional steps per neuron removed; most recently, `d_ff=9` achieved 100% accuracy at 20,000 steps, motivating the next boundary test with the same convergence allowance.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=9)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=20000)
=======
    p.add_argument("--d-ff", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=24000)
>>>>>>> REPLACE