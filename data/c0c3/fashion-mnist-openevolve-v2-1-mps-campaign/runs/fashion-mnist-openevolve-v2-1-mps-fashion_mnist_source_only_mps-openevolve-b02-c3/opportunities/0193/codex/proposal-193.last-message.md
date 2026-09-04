MECHANISM: Three-point quadratic crop-power refinement

HYPOTHESIS: Setting the crop fusion power to 1.3509 will preserve 9,349 correct predictions while reducing validation cross-entropy below 0.1876555103302002.

INTENDED_EDIT: Move the crop-consensus power from 1.3515 to the quadratic minimum fitted from the three nearest verified powers, preserving training, crop weights, orientation fusion, and calibration.

EVIDENCE: Powers 1.34, 1.3477, and 1.3515 produced cross-entropies 0.1876555618286133, 0.1876555145263672, and 0.1876555103302002 respectively, with all relevant endpoints retaining 9,349 correct; fitting these measurements places the local minimum near 1.3509.

<<<<<<< SEARCH
        fusion_power = 1.3515
=======
        fusion_power = 1.3509
>>>>>>> REPLACE