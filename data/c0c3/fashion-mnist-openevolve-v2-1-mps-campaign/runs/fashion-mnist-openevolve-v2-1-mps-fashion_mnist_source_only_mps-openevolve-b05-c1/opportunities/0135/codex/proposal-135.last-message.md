MECHANISM: Conservative center-view boundary refinement

HYPOTHESIS: A center weight of 1.81731414794921875 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015732.

INTENDED_EDIT: Halve the latest attempted downward adjustment to the center-view ensemble weight, preserving all views and normalization.

EVIDENCE: Weight 1.81732177734375 retained 9,360 correct, whereas 1.81689453125 lost one prediction despite lower cross-entropy; the intervening 1.8173065185546875 run timed out, so a smaller step tests the established cross-entropy trend with less boundary risk.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
=======
        center_weight = 1.81731414794921875
>>>>>>> REPLACE