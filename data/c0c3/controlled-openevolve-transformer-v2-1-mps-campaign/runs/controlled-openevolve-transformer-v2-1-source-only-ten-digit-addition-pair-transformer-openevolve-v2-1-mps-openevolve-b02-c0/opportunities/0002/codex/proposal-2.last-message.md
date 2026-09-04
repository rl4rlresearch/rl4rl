MECHANISM: Single-neuron feed-forward bottleneck ablation

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower the parameter count from 1,644 to 1,627 while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one feed-forward hidden unit while preserving the successful attention architecture and training procedure.

EVIDENCE: `d_ff=12` achieved 99.96% accuracy, whereas `d_ff=8` fell to 63.99%; this suggests a sharp capacity threshold and motivates the smallest possible width reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE