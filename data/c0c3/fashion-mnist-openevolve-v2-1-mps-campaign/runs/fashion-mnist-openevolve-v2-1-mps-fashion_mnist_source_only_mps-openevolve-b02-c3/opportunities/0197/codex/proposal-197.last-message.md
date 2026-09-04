MECHANISM: Accuracy-directed stronger crop-consensus fusion

HYPOTHESIS: Raising the crop fusion power to 1.50 will exceed 9,349 correct predictions by crossing additional beneficial decision boundaries beyond the one observed between powers 1.174 and 1.34.

INTENDED_EDIT: Increase only the crop-consensus power, preserving the verified architecture, training procedure, crop weights, orientation averaging, and calibration.

EVIDENCE: Increasing fusion power from 1.174 to 1.34 raised validation_correct from 9,348 to 9,349, while all tested values through 1.3515 retained that gain; because correctness dominates cross-entropy, the unexplored higher-power direction is more informative than further local calibration refinement.

<<<<<<< SEARCH
        fusion_power = 1.3515
=======
        fusion_power = 1.50
>>>>>>> REPLACE