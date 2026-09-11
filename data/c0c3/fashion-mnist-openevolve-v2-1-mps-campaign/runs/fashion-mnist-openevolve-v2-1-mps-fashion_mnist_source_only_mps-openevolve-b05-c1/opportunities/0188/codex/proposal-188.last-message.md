MECHANISM: Low-consensus confidence restoration

HYPOTHESIS: A 0.02 confidence bonus for below-average TTA agreement will preserve exactly 9,360 predictions while lowering cross-entropy below 0.18569574165344238.

INTENDED_EDIT: Add a small positive centered-hinge term to the existing agreement-conditioned confidence scale.

EVIDENCE: An additional low-agreement confidence penalty increased cross-entropy to 0.1856981330871582 without changing predictions, indicating that the remaining low-consensus tail was over-attenuated and motivating a conservative perturbation in the opposite direction.

<<<<<<< SEARCH
        confidence_scale = torch.exp(0.31 * agreement)
=======
        confidence_scale = torch.exp(
            0.31 * agreement + 0.02 * F.relu(-agreement)
        )
>>>>>>> REPLACE