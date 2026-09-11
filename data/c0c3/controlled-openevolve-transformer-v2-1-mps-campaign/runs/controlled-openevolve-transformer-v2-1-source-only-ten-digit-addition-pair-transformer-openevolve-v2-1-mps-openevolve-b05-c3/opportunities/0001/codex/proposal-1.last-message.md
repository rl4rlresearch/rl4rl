MECHANISM: Narrower nonlinear feature bottleneck

HYPOTHESIS: Reducing the single block’s feed-forward width from 12 to 10 will lower deduplicated parameters from 1644 to 1610 while retaining at least 99% accuracy, given the current model’s 99.96% accuracy margin.

INTENDED_EDIT: Set the default feed-forward width to 10 and update run labels accordingly; all training and decoding behavior remains unchanged.

EVIDENCE: The d_model=8, d_ff=12 design achieved 0.9996 accuracy at 1644 parameters after 4999 steps, indicating enough margin to test a modest two-unit reduction in the MLP bottleneck.

<<<<<<< SEARCH
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff12")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff12"))
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--best-ckpt-out", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--last-ckpt-out", type=Path, default=Path("checkpoints/last.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)

    # model
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--run-name", type=str, default="repro_l1_d8_ff10")
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/repro_l1_d8_ff10"))
    p.add_argument("--split-dir", type=Path, default=Path("results/data"))
    p.add_argument("--best-ckpt-out", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--last-ckpt-out", type=Path, default=Path("checkpoints/last.pt"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=123)

    # model
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE