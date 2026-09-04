MECHANISM: Split-slope TTA-agreement calibration

HYPOTHESIS: Increasing the confidence slope only for above-average-agreement examples by 0.014 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18569544525146484.

INTENDED_EDIT: Add a positive centered-hinge term for above-average TTA agreement, complementing the verified low-agreement slope reduction.

EVIDENCE: The globally fitted agreement slope was optimal near 0.31, yet reducing only the low-agreement slope to 0.296 improved cross-entropy. At the former joint optimum, that directional improvement implies the high-agreement branch favored a larger slope; positive per-example scaling cannot change argmax predictions.

<<<<<<< SEARCH
        confidence_scale = torch.exp(
            0.31 * agreement + 0.014 * F.relu(-agreement)
        )
=======
        confidence_scale = torch.exp(
            0.31 * agreement
            + 0.014 * F.relu(-agreement)
            + 0.014 * F.relu(agreement)
        )
>>>>>>> REPLACE