MECHANISM: Narrow two-stage transformer

HYPOTHESIS: Replacing the single width-8 block with two width-6 blocks will produce 1518 learned parameters and maintain at least 99% accuracy, because depth permits attention routing and nonlinear refinement to occur in separate learned stages while preserving two attention heads and 12 MLP units per stage.

INTENDED_EDIT: Change the default architecture from one width-8 block to two width-6 blocks, retaining two heads, the full 12-unit MLP width, training procedure, and protected decoding interface.

EVIDENCE: Repeated 1599-parameter scalar ablations of the qualified one-block architecture failed, showing that further coordinate shaving within that computation is brittle. The 1600-parameter model nevertheless demonstrates that the task fits near this budget, motivating a different allocation: narrower token and position representations fund a second complete learned attention/MLP stage at 1518 parameters.

<<<<<<< SEARCH
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--d-model", type=int, default=6)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=12)
>>>>>>> REPLACE