MECHANISM: Narrower nonlinear residual bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will remove 68 learned parameters (1644 → 1576) while retaining at least 99% accuracy because the current 99.96% result provides substantial error-margin evidence that the MLP is overprovisioned.

INTENDED_EDIT: Change the default feed-forward width from 12 to 8.

EVIDENCE: The current one-layer, width-8 model achieved 0.9996 accuracy with 1644 parameters; this motivates isolating a moderate 33% reduction in its MLP width before changing attention or embedding capacity.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE