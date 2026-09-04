MECHANISM: Separable 3×3 translation TTA

HYPOTHESIS: Adding diagonally shifted views with separable product weights will exceed 9,348 correct predictions by reducing residual two-axis alignment sensitivity while preserving the best verified calibration.

INTENDED_EDIT: Restore Reference Design 2’s equal orientation weighting and temperature, then extend its axial crop ensemble to all nine one-pixel translations; derive each diagonal weight as the product of its vertical and horizontal axial weights divided by the center weight.

EVIDENCE: Axial translation TTA increased correctness from 9,344 to 9,348, whereas subsequent temperature and micro-weight probes only changed cross-entropy. A separable 3×3 ensemble is the smallest untested extension of that successful mechanism and adds diagonal alignment coverage without changing training or parameters.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
        native_weight = 0.99999988079071044921875
        reflected_weight = 1.00000011920928955078125
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + (
                weight * native_weight
            ) * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + (
                weight * reflected_weight
            ) * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (
            (native_weight + reflected_weight) * sum(crop_weights)
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
=======
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        crops = (
            padded[:, :, 1:29, 1:29],
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
            padded[:, :, 0:28, 0:28],
            padded[:, :, 0:28, 2:30],
            padded[:, :, 2:30, 0:28],
            padded[:, :, 2:30, 2:30],
        )
        probabilities = torch.zeros(
            (images.shape[0], 10),
            device=images.device,
            dtype=images.dtype,
        )
        up_weight = 1.3120136260986328125
        down_weight = 1.3149394989013671875
        left_weight = 0.686523377895355224609375
        right_weight = 0.686523497104644775390625
        crop_weights = (
            3.0,
            up_weight,
            down_weight,
            left_weight,
            right_weight,
            up_weight * left_weight / 3.0,
            up_weight * right_weight / 3.0,
            down_weight * left_weight / 3.0,
            down_weight * right_weight / 3.0,
        )
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE