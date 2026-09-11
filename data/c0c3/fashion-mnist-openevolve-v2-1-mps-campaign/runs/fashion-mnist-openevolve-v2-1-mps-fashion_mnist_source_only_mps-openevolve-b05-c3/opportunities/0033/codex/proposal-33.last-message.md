MECHANISM: Equal-step extension of inference-aligned augmentation

HYPOTHESIS: Starting cardinal augmentation at 19/32 of training will exceed 9,167 correct predictions by extending the beneficial terminal phase by the same interval that previously improved 9,163 to 9,167.

INTENDED_EDIT: Move the broad-to-cardinal augmentation transition from 39/64 to 19/32 while preserving all other validated settings.

EVIDENCE: Earlier transitions improved correct predictions monotonically from 9,141 at 3/4 to 9,155 at 2/3, 9,163 at 5/8, and 9,167 at 39/64; the intervening timeouts supplied no contrary accuracy result.

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE