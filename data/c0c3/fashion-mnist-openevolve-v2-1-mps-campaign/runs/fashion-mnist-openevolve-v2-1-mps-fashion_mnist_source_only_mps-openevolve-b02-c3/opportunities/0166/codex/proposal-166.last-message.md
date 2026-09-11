MECHANISM: Log-domain orientation consensus

HYPOTHESIS: Replacing arithmetic native/reflected probability averaging with normalized geometric averaging will exceed 9,348 correct predictions by suppressing class evidence that is strong in only one orientation while preserving the verified crop weighting and power-mean fusion.

INTENDED_EDIT: Average each crop’s native and reflected logits before softmax, then retain the existing 1.174-order crop fusion and calibrated temperature.

EVIDENCE: Hierarchical orientation-before-crop consensus retained 9,348 correct while improving cross-entropy over flat fusion, indicating that resolving orientation disagreement before crop aggregation is beneficial; architecture changes, diagonal crops, and altered training translations all reduced correctness.

<<<<<<< SEARCH
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
=======
        for crop, weight in zip(crops, crop_weights):
            native_logits = self._forward_once(crop)
            reflected_logits = self._forward_once(crop.flip(-1))
            orientation_consensus = F.softmax(
                0.5 * (native_logits + reflected_logits),
                dim=1,
            )
            powered_probabilities = powered_probabilities + (
                2.0 * weight
            ) * orientation_consensus.pow(fusion_power)
>>>>>>> REPLACE