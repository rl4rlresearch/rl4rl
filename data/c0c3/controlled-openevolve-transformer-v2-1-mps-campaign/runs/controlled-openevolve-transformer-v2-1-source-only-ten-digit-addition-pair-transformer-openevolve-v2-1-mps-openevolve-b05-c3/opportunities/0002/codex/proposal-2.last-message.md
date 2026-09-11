MECHANISM: Single-unit MLP bottleneck reduction

HYPOTHESIS: An intermediate d_ff=11 model will reduce parameters from 1644 to 1627 while retaining at least 99% accuracy, testing whether the failure at d_ff=10 marks a sharp capacity threshold.

INTENDED_EDIT: Reduce the default feed-forward width from 12 to 11 and update run labels accordingly.

EVIDENCE: The d_ff=12 design reached 99.96% accuracy, while d_ff=10 reached only 17.93%; d_ff=11 is the most informative untested point between these outcomes.

<<<<<<< SEARCH
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
=======
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff11")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff11"))
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE