MECHANISM: Upper-interval crop-power boundary probe

HYPOTHESIS: Setting crop fusion power to 1.42575 will exceed 9,349 correct predictions if another beneficial decision boundary occurs before the regression observed at power 1.50.

INTENDED_EDIT: Move crop-consensus power halfway between the best 1.3515 setting and the accuracy-regressing 1.50 endpoint, preserving all other behavior.

EVIDENCE: Raising power from 1.174 to 1.34 gained one correct prediction, while 1.50 lost it; probing the untested midpoint is the most informative accuracy-directed test within that transition interval.

<<<<<<< SEARCH
        fusion_power = 1.3515
=======
        fusion_power = 1.42575
>>>>>>> REPLACE