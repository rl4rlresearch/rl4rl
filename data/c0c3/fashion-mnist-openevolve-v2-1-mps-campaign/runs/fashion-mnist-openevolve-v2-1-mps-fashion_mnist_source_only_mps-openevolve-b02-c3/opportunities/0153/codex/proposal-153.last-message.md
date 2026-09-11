MECHANISM: Hierarchical orientation-then-crop power-mean fusion

HYPOTHESIS: Averaging each native/reflected pair before applying the 1.174-order crop power mean will exceed 9,348 correct predictions by suppressing orientation-specific confidence outliers while retaining the beneficial emphasis on crop-specific evidence.

INTENDED_EDIT: Replace the flat power mean over all ten views with arithmetic fusion within each reflection pair followed by the verified higher-order fusion across the five axial crops.

EVIDENCE: The 1.174-order flat power mean achieved the best cross-entropy with 9,348 correct, while favoring either orientation and reflection-consistency training both failed; this suggests equal orientation consensus should be preserved while isolating the useful higher-order weighting to crop disagreement.

<<<<<<< SEARCH
        for crop, weight in zip(crops, crop_weights):
            native_probabilities = F.softmax(
                self._forward_once(crop), dim=1
            )
            reflected_probabilities = F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
            powered_probabilities = powered_probabilities + weight * (
                native_probabilities.pow(fusion_power)
                + reflected_probabilities.pow(fusion_power)
            )
=======
        for crop, weight in zip(crops, crop_weights):
            native_probabilities = F.softmax(
                self._forward_once(crop), dim=1
            )
            reflected_probabilities = F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
            orientation_consensus = 0.5 * (
                native_probabilities + reflected_probabilities
            )
            powered_probabilities = powered_probabilities + (
                2.0 * weight
            ) * orientation_consensus.pow(fusion_power)
>>>>>>> REPLACE