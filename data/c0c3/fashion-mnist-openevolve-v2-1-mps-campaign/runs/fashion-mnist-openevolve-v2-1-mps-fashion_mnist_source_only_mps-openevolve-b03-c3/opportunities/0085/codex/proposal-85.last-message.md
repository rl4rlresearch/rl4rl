MECHANISM: Conservative outer-shift axial interpolation

HYPOTHESIS: Moving one quarter of the way from the best 0.009375/0.00625/0.003125 radius-2 weighting toward the stronger axial weighting will preserve 9,268 correct predictions while reducing validation cross-entropy below 0.2124557217.

INTENDED_EDIT: Increase each radius-2 axial-view weight to 0.01015625 and decrease each corner-view weight to 0.00234375, preserving the total ensemble weight and all training settings.

EVIDENCE: Mild axial redistribution preserved 9,268 correct and improved cross-entropy over uniform weighting, while full corner-to-axial redistribution further improved cross-entropy but lost one prediction; a conservative interpolation tests the interval between them.

<<<<<<< SEARCH
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
=======
                    if manhattan_distance == 2:
                        weight = 0.01015625
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.00234375
>>>>>>> REPLACE