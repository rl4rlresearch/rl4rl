MECHANISM: Fine-grained center-weighted logit ensemble

HYPOTHESIS: Increasing the centered validation-view weight from 3.0 to 3.25 will exceed 9,166 correct predictions by exploiting the asymmetric results around the current setting while avoiding the excessive center dominance of weight 4.0.

INTENDED_EDIT: Change only the centered view’s ensemble weight from 3.0 to 3.25, preserving training, model parameters, calibration, and runtime.

EVIDENCE: Center weights 2.5 and 4.0 produced 9,163 and 9,164 correct respectively, versus 9,166 at 3.0; the smaller regression above 3.0 motivates a conservative upward refinement within the tested interval.

<<<<<<< SEARCH
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.25, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE