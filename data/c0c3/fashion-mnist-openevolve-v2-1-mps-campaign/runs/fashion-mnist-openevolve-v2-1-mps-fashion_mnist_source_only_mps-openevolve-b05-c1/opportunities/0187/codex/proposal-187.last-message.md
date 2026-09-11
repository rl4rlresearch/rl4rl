MECHANISM: Low-consensus tail confidence calibration

HYPOTHESIS: Applying a small additional confidence penalty only to below-average TTA agreement will preserve exactly 9,360 argmax predictions while reducing validation cross-entropy below 0.18569574165344238.

INTENDED_EDIT: Add a centered hinge feature for below-average agreement and give it a conservative negative coefficient in the existing positive confidence scale.

EVIDENCE: Linear agreement calibration reached its fitted optimum at 0.31 without changing any predictions, while consistently improving cross-entropy; this motivates testing whether the remaining calibration error is asymmetric in the low-consensus tail.

<<<<<<< SEARCH
        agreement = agreement - agreement.mean()
        confidence_scale = torch.exp(0.31 * agreement)
=======
        agreement = agreement - agreement.mean()
        low_consensus = F.relu(-agreement)
        low_consensus = low_consensus - low_consensus.mean()
        confidence_scale = torch.exp(
            0.31 * agreement - 0.04 * low_consensus
        )
>>>>>>> REPLACE