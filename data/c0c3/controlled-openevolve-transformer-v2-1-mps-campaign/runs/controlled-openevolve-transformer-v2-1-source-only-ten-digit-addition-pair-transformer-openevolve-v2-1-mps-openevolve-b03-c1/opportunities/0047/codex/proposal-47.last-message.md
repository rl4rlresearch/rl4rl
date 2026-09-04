MECHANISM: One-unit MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower the model from 1,209 to 1,194 learned parameters while retaining at least 99% accuracy, because the successful zero-mean MLP output compression indicates remaining MLP redundancy, whereas the failed rank-five lexical bottleneck shows lexical rank is the riskier capacity target.

INTENDED_EDIT: Change the default feed-forward width from 12 hidden units to 11, preserving the architecture, training schedule, and learned causal attention pathway.

EVIDENCE: The 1,209-parameter model retained 99.89% accuracy after removing 12 MLP output-weight parameters, supporting a small additional MLP reduction; by contrast, reducing lexical rank caused accuracy to collapse to 37.87%.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE