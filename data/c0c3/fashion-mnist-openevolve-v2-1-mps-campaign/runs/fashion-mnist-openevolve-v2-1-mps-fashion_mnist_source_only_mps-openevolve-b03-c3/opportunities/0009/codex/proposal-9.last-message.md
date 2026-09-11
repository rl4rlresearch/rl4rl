MECHANISM: Flip-orbit-complete supervised minibatches

HYPOTHESIS: Restoring the 9,155-correct weighted-translation/logit-ensemble design and supervising both horizontal orientations of every sampled crop will exceed 9,155 correct predictions by eliminating flip-sampling variance and aligning training with the proven flip-averaged evaluation.

INTENDED_EDIT: Use reliability-weighted translation sampling, restore weighted logit averaging at evaluation, and train each sampled crop together with its horizontal flip in one forward pass.

EVIDENCE: Reference Design 2 achieved 9,155 correct with weighted translation sampling and logit aggregation, while probability aggregation reduced performance to 9,047; flip averaging is already proven at evaluation, motivating complete paired flip exposure during training.

<<<<<<< SEARCH
    def _flip_probability_average(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images).softmax(dim=-1)
            + self._forward_once(images.flip(-1)).softmax(dim=-1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        probabilities = 0.36 * self._flip_probability_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
                probabilities = probabilities + (
                    weight * self._flip_probability_average(shifted)
                )

        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()
=======
    def _flip_average(self, images: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            self._forward_once(images)
            + self._forward_once(images.flip(-1))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        logits = 0.36 * self._flip_average(images)

        for delta_y in range(-2, 3):
            for delta_x in range(-2, 3):
                if delta_y == 0 and delta_x == 0:
                    continue

                shifted = padded[
                    :,
                    :,
                    2 + delta_y : 2 + delta_y + height,
                    2 + delta_x : 2 + delta_x + width,
                ]
                radius = max(abs(delta_y), abs(delta_x))
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
                logits = logits + weight * self._flip_average(shifted)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offsets = torch.randint(0, 5, (images.size(0), 2), device=images.device)
=======
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    translation_weights = images.new_tensor(
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
    )
    flat_offsets = torch.multinomial(
        translation_weights, images.size(0), replacement=True
    )
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    flip_mask = torch.rand(images.size(0), device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None], images.flip(-1), images
    )
    return images, labels
=======
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images), paired_labels, label_smoothing=0.03
    )
>>>>>>> REPLACE