MECHANISM: Narrower nonlinear channel

HYPOTHESIS: Reducing the sole MLP width from 12 to 8 will retain at least 99% accuracy while reducing parameters from 1,644 to 1,576.

INTENDED_EDIT: Set the default feed-forward width to 8; all training and optimization settings remain unchanged.

EVIDENCE: The current width-12 model achieved 99.96% accuracy, leaving substantial margin above the 99% requirement; removing four MLP units saves 68 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE