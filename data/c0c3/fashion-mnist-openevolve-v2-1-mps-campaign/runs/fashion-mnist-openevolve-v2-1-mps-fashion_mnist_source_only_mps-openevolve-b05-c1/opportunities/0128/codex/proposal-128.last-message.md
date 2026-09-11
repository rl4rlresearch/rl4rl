MECHANISM: Fine-grained center-prior boundary search

HYPOTHESIS: A center weight of 1.81689453125 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015785.

INTENDED_EDIT: Halve the remaining interval from the verified 1.8173828125 center weight toward the unresolved 1.81640625 candidate, preserving all evaluation views and normalization.

EVIDENCE: Every verified reduction from 1.875 through 1.8173828125 retained 9,360 correct while monotonically lowering cross-entropy; the 1.81640625 run timed out and therefore provides no contrary accuracy evidence.

<<<<<<< SEARCH
        center_weight = 1.8173828125
=======
        center_weight = 1.81689453125
>>>>>>> REPLACE