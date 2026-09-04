MECHANISM: Local warmdown-duration interpolation

HYPOTHESIS: A 78% linear warmdown will reduce val_bpb below 0.984506 by matching the approximately 78.4% minimum implied by the measured 70%, 80%, and 84% schedules.

INTENDED_EDIT: Preserve the winning architecture, batch size, optimizer, and linear schedule while beginning learning-rate decay at 22% rather than 20% of the training window.

EVIDENCE: Warmdowns of 70%, 80%, and 84% achieved val_bpb values of 0.984948, 0.984506, and 0.984696; quadratic interpolation of these nearest measurements places the local minimum near 78.4%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.8    # begin linear LR decay after 20% of the time budget
=======
WARMDOWN_RATIO = 0.78   # begin linear LR decay after 22% of the time budget
>>>>>>> REPLACE