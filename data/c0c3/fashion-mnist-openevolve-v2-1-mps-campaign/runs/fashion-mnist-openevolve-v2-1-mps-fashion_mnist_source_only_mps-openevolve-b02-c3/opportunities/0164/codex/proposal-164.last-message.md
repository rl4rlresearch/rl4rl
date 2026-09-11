MECHANISM: Evaluation-aligned anisotropic translation augmentation

HYPOTHESIS: Training on axial translations with probabilities proportional to the best verified crop weights will exceed 9,348 correct predictions by emphasizing useful vertical invariance while reducing weaker horizontal perturbations.

INTENDED_EDIT: Restore hierarchical orientation-then-crop power-mean evaluation at the best verified temperature, and align training’s axial translation distribution with its crop weights.

EVIDENCE: Hierarchical fusion preserved 9,348 correct while improving cross-entropy to 0.18766820449829102; its optimized weights favor vertical over horizontal crops, while adding diagonal views reduced correctness to 9,335.

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
            orientation_consensus = 0.5 * (
                native_probabilities + reflected_probabilities
            )
            powered_probabilities = powered_probabilities + (
                2.0 * weight
            ) * orientation_consensus.pow(fusion_power)
        probabilities = (
            powered_probabilities / (2.0 * sum(crop_weights))
        ).pow(1.0 / fusion_power)
        probabilities = probabilities / probabilities.sum(
            dim=1,
            keepdim=True,
        )
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753169953823089599609375
>>>>>>> REPLACE

<<<<<<< SEARCH
    translation = torch.randint(0, 7, (batch,), device=images.device)
    offsets_y = torch.tensor(
        (1, 1, 1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
=======
    translation = torch.multinomial(
        torch.tensor(
            (
                3.0,
                1.3120136260986328125,
                1.3149394989013671875,
                0.686523377895355224609375,
                0.686523497104644775390625,
            ),
            device=images.device,
        ),
        batch,
        replacement=True,
    )
    offsets_y = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[translation].unsqueeze(1)
    offsets_x = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[translation].unsqueeze(1)
>>>>>>> REPLACE