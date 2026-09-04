MECHANISM: Half-strength super-arithmetic orientation fusion

HYPOTHESIS: Using orientation power 1.087—halfway between arithmetic averaging and the accuracy-preserving 1.174 endpoint—will exceed 9,348 correct predictions if beneficial decision-boundary crossings occur before offsetting regressions.

INTENDED_EDIT: Replace arithmetic native/reflected probability averaging with a normalized 1.087-order power mean while retaining the verified crop fusion, weights, and reciprocal calibration.

EVIDENCE: Arithmetic orientation fusion and 1.174-order orientation fusion both achieved 9,348 correct, while the latter only modestly worsened cross-entropy; testing their midpoint is an informative probe for differently ordered discrete prediction changes.

<<<<<<< SEARCH
        fusion_power = 1.174
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
=======
        fusion_power = 1.174
        orientation_power = 1.087
        for crop, weight in zip(crops, crop_weights):
            native_probabilities = F.softmax(
                self._forward_once(crop), dim=1
            )
            reflected_probabilities = F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
            orientation_consensus = (
                0.5
                * (
                    native_probabilities.pow(orientation_power)
                    + reflected_probabilities.pow(orientation_power)
                )
            ).pow(1.0 / orientation_power)
            orientation_consensus = orientation_consensus / (
                orientation_consensus.sum(dim=1, keepdim=True)
            )
>>>>>>> REPLACE