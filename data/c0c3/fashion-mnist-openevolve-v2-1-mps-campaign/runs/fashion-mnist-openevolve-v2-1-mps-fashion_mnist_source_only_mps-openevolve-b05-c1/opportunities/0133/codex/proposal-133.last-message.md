MECHANISM: Conservative classification-boundary refinement

HYPOTHESIS: A center weight of 1.817291259765625 will retain 9,360 correct predictions while reducing validation cross-entropy below 0.1860015732.

INTENDED_EDIT: Halve the previous successful downward step in center-view weight, preserving all evaluation views and normalization.

EVIDENCE: Reducing the weight from 1.8173828125 to 1.81732177734375 retained 9,360 correct and lowered cross-entropy, while 1.81689453125 lost one prediction; a half-sized continuation exploits the monotonic cross-entropy trend with less boundary risk.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
=======
        center_weight = 1.817291259765625
>>>>>>> REPLACE