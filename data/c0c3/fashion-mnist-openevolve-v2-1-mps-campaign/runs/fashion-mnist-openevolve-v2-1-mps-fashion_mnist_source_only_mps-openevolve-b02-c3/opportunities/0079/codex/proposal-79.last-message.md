MECHANISM: Near-optimal center-weight boundary refinement

HYPOTHESIS: A centered-crop weight of 2.9375 will retain 9,344 correct predictions while lowering cross-entropy below the weight-3.0 result.

INTENDED_EDIT: Move the centered TTA crop weight halfway between the 2.875 and 3.0 settings, retaining the verified 0.98 EMA.

EVIDENCE: Weight 3.0 achieved 9,344 correct, while weight 2.875 achieved 9,343 with lower cross-entropy; their untested midpoint directly refines the accuracy boundary and can improve the tie-breaker without changing training.

<<<<<<< SEARCH
        crop_weights = (2.5, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (2.9375, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE