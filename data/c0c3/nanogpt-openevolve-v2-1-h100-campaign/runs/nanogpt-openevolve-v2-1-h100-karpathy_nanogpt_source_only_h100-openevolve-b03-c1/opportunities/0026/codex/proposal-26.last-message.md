MECHANISM: Local warmdown-duration response-surface refinement

HYPOTHESIS: A 76% linear warmdown will reduce val_bpb below 0.984312 by approaching the approximately 76.3% minimum implied by the measured 70%, 78%, and 80% schedules.

INTENDED_EDIT: Preserve the winning architecture, batch size, optimizer, and linear schedule while beginning learning-rate decay at 24% rather than 22% of the training window.

EVIDENCE: Warmdowns of 70%, 78%, and 80% achieved val_bpb values of 0.984948, 0.984312, and 0.984506; quadratic interpolation of these nearest measurements places the estimated minimum near 76.3%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.78   # begin linear LR decay after 22% of the time budget
=======
WARMDOWN_RATIO = 0.76   # begin linear LR decay after 24% of the time budget
>>>>>>> REPLACE