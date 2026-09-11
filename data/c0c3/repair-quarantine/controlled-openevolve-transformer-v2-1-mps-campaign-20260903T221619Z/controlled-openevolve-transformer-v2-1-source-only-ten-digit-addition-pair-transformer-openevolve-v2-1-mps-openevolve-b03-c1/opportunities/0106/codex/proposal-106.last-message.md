MECHANISM: One-unit MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 10 to 9 will lower the verified model from 606 to 591 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one hidden unit from the transformer MLP while preserving the verified lexical representation and attention architecture.

EVIDENCE: The prior `d_ff` reduction from 11 to 10 retained 99.22% accuracy, and the current width-10 model reaches 99.89%, making another single-unit reduction the most direct test of remaining MLP capacity.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=10)
=======
    p.add_argument("--d-ff", type=int, default=9)
>>>>>>> REPLACE