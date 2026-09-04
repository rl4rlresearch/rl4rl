MECHANISM: One-unit MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 11 to 10 will lower the model from 1,194 to 1,179 learned parameters while retaining at least 99% accuracy, because the preceding 12-to-11 reduction achieved 99.98% accuracy and showed substantial remaining MLP capacity.

INTENDED_EDIT: Change the default feed-forward width from 11 hidden units to 10 while preserving all other architecture and training settings.

EVIDENCE: The verified 1,194-parameter design with `d_ff=11` reached 99.98% accuracy, improving on the already successful 1,209-parameter `d_ff=12` design; this supports another conservative one-unit reduction, while the failed rank-five lexical bottleneck indicates lexical compression is substantially riskier.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE