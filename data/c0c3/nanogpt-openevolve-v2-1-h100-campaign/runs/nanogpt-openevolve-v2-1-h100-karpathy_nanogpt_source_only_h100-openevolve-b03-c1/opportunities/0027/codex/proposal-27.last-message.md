MECHANISM: Local warmdown-duration response-surface refinement

HYPOTHESIS: A 78.25% linear warmdown will reduce val_bpb below 0.984312 by matching the approximately 78.24% minimum implied by the measured 76%, 78%, and 80% schedules.

INTENDED_EDIT: Preserve the winning architecture, batch size, optimizer, and linear schedule while beginning learning-rate decay at 21.75% rather than 22% of the training window.

EVIDENCE: Warmdowns of 76%, 78%, and 80% produced val_bpb values of 0.984631, 0.984312, and 0.984506; quadratic interpolation of these equally spaced measurements places the local minimum near 78.24%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.78   # begin linear LR decay after 22% of the time budget
=======
WARMDOWN_RATIO = 0.7825 # begin linear LR decay after 21.75% of the time budget
>>>>>>> REPLACE