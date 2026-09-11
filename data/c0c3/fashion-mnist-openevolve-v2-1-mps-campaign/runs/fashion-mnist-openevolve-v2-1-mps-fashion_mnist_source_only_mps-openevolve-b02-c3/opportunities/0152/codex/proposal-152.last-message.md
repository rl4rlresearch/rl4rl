MECHANISM: Reflection-pair Jensen–Shannon consistency regularization

HYPOTHESIS: Adding a 0.10-weight Jensen–Shannon penalty between paired native and reflected predictions to the verified 1.174-order fusion design will exceed 9,348 correct predictions by reducing orientation-sensitive errors.

INTENDED_EDIT: Restore Reference Design 2’s best verified power-mean evaluation and augment the existing paired supervised loss with symmetric reflection-consistency regularization.

EVIDENCE: The 1.174-order fusion achieved the best verified cross-entropy with 9,348 correct, while preferring either orientation was not an improvement; this motivates teaching the model itself to agree across the equally weighted reflection pair.

<<<<<<< SEARCH
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
        for crop, weight in zip(crops, crop_weights):
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop), dim=1
            )
            probabilities = probabilities + weight * F.softmax(
                self._forward_once(crop.flip(-1)), dim=1
            )
        probabilities = probabilities / (2.0 * sum(crop_weights))
        log_probabilities = probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
        lower_temperature_logits = (
            log_probabilities / 0.753170073032379150390625
        )
        higher_temperature_logits = (
            log_probabilities / 0.75317013263702392578125
        )
        return torch.lerp(
            lower_temperature_logits,
            higher_temperature_logits,
            0.25,
        )
=======
        powered_probabilities = torch.zeros(
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
        fusion_power = 1.174
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
        probabilities = (
            powered_probabilities / (2.0 * sum(crop_weights))
        ).pow(1.0 / fusion_power)
        probabilities = probabilities / probabilities.sum(
            dim=1,
            keepdim=True,
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
=======
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    supervised_loss = F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )

    native_logits, reflected_logits = paired_logits.chunk(2, dim=0)
    native_log_probabilities = F.log_softmax(native_logits, dim=1)
    reflected_log_probabilities = F.log_softmax(
        reflected_logits,
        dim=1,
    )
    consensus_log_probabilities = torch.logaddexp(
        native_log_probabilities,
        reflected_log_probabilities,
    ) - math.log(2.0)
    consistency_loss = 0.5 * (
        (
            native_log_probabilities.exp()
            * (
                native_log_probabilities
                - consensus_log_probabilities
            )
        ).sum(dim=1)
        + (
            reflected_log_probabilities.exp()
            * (
                reflected_log_probabilities
                - consensus_log_probabilities
            )
        ).sum(dim=1)
    ).mean()
    return supervised_loss + 0.10 * consistency_loss
>>>>>>> REPLACE