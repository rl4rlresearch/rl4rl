MECHANISM: Local warmdown-duration interpolation

HYPOTHESIS: A linear warmdown lasting 84% of training will reduce val_bpb below 0.984506 by approaching the approximately 83.5% optimum implied by the measured 70%, 80%, and 90% results.

INTENDED_EDIT: Preserve the winning architecture, 112-sequence batch, optimizer, and linear schedule while beginning learning-rate decay at 16% rather than 20% of the training window.

EVIDENCE: Linear warmdowns of 70%, 80%, and 90% achieved val_bpb values of 0.984948, 0.984506, and 0.984582; quadratic interpolation places the local minimum near 83.5%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.8    # begin linear LR decay after 20% of the time budget
=======
WARMDOWN_RATIO = 0.84   # begin linear LR decay after 16% of the time budget
>>>>>>> REPLACE