MECHANISM: Fine-grained center-prior boundary search

HYPOTHESIS: A center weight of 1.8173828125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860016449.

INTENDED_EDIT: Halve the step from the verified 1.818359375 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.

EVIDENCE: Reductions from 1.875 through 1.84375, 1.828125, 1.8203125, and 1.818359375 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 verification timed out and supplied no contrary accuracy evidence.

<<<<<<< SEARCH
        center_weight = 1.818359375
=======
        center_weight = 1.8173828125
>>>>>>> REPLACE