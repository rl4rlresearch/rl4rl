MECHANISM: Conservative extension of inference-aligned augmentation

HYPOTHESIS: Beginning center/cardinal augmentation after three-fifths of training will exceed 9,167 correct predictions by extending the consistently beneficial terminal phase without sharply reducing early broad-translation regularization.

INTENDED_EDIT: Move the broad-to-cardinal augmentation transition from two-thirds to three-fifths of training while preserving all other validated settings.

EVIDENCE: Earlier transitions monotonically improved validation correct from 9,141 at three-quarters to 9,155 at two-thirds, 9,163 at five-eighths, and 9,167 at 39/64; the timed-out still-earlier trials supplied no contrary accuracy evidence.

<<<<<<< SEARCH
    if step * 3 < total_steps * 2:
=======
    if step * 5 < total_steps * 3:
>>>>>>> REPLACE