MECHANISM: Narrower feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 10 will lower the model from 1,644 to 1,610 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Change the default feed-forward width to 10; keep architecture, training schedule, attention, and decoding unchanged.

EVIDENCE: The current 1,644-parameter model reaches 99.96% accuracy, leaving a 0.96-point margin above the threshold and motivating a conservative 34-parameter reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE