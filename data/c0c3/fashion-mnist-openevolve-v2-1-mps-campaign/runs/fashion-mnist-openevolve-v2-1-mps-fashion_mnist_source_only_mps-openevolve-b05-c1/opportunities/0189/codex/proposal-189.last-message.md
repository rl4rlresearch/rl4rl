MECHANISM: Incremental low-consensus confidence restoration

HYPOTHESIS: Increasing the below-average-agreement confidence bonus from 0.02 to 0.04 will preserve exactly 9,360 predictions while lowering validation cross-entropy below 0.18569551315307617.

INTENDED_EDIT: Increase only the positive centered-hinge coefficient in the existing TTA-agreement confidence scale.

EVIDENCE: A confidence penalty for low-agreement examples worsened cross-entropy to 0.1856981330871582, while the opposite 0.02 bonus improved it to 0.18569551315307617 without changing any predictions, motivating one further equal-sized step in the beneficial direction.

<<<<<<< SEARCH
        confidence_scale = torch.exp(
            0.31 * agreement + 0.02 * F.relu(-agreement)
        )
=======
        confidence_scale = torch.exp(
            0.31 * agreement + 0.04 * F.relu(-agreement)
        )
>>>>>>> REPLACE