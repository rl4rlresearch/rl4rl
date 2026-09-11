MECHANISM: Differential-basis contour encoding

HYPOTHESIS: Replacing the shared raw-pixel-only front end with raw, local-contrast, signed-gradient, and edge-energy channels—while retaining the 9,186-correct training and evaluation recipe—will exceed 9,186 correct predictions by exposing shape evidence directly during the limited 522 optimizer steps.

INTENDED_EDIT: Challenge the assumption that another raw-pixel CNN refinement is the best use of capacity: learn over a fixed five-channel differential image basis, restore reliability-weighted translation sampling and 5×5 logit aggregation, and supervise both flip orientations.

EVIDENCE: Flip-complete weighted-translation training reached 9,186 correct, whereas adding another same-type residual refinement reached only 9,039; this favors retaining the proven invariance recipe while testing a genuinely different input representation rather than adding more conventional depth.

<<<<<<< SEARCH
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    @staticmethod
    def _image_basis(images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        local_mean = F.avg_pool2d(padded, kernel_size=3, stride=1)
        local_contrast = images - local_mean

        gradient_x = 0.125 * (
            padded[:, :, :-2, 2:]
            + 2.0 * padded[:, :, 1:-1, 2:]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, 1:-1, :-2]
            - padded[:, :, 2:, :-2]
        )
        gradient_y = 0.125 * (
            padded[:, :, 2:, :-2]
            + 2.0 * padded[:, :, 2:, 1:-1]
            + padded[:, :, 2:, 2:]
            - padded[:, :, :-2, :-2]
            - 2.0 * padded[:, :, :-2, 1:-1]
            - padded[:, :, :-2, 2:]
        )
        edge_energy = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1.0e-6
        )
        return torch.cat(
            (
                images,
                local_contrast,
                gradient_x,
                gradient_y,
                edge_energy,
            ),
            dim=1,
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        cardinal_views = (
            padded[:, :, :height, 1 : width + 1],
            padded[:, :, 2 : height + 2, 1 : width + 1],
            padded[:, :, 1 : height + 1, :width],
            padded[:, :, 1 : height + 1, 2 : width + 2],
        )
        diagonal_views = (
            padded[:, :, :height, :width],
            padded[:, :, :height, 2 : width + 2],
            padded[:, :, 2 : height + 2, :width],
            padded[:, :, 2 : height + 2, 2 : width + 2],
        )

        logits = 0.4 * self._flip_average(images)
        for shifted in cardinal_views:
            logits = logits + 0.1 * self._flip_average(shifted)
        for shifted in diagonal_views:
            logits = logits + 0.05 * self._flip_average(shifted)
        return logits
=======
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
    height, width = images.shape[-2:]
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    offsets = torch.randint(0, 5, (images.size(0), 2), device=images.device)

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    row_index = rows[:, None, :, None].expand(
        -1, images.size(1), -1, padded.size(3)
    )
    images = padded.gather(2, row_index)

    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    column_index = columns[:, None, None, :].expand(
        -1, images.size(1), height, -1
    )
    images = images.gather(3, column_index)

    flip_mask = torch.rand(images.size(0), device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None], images.flip(-1), images
    )
    return images, labels
=======
    height, width = images.shape[-2:]
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

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
    row_index = rows[:, None, :, None].expand(
        -1, images.size(1), -1, padded.size(3)
    )
    images = padded.gather(2, row_index)

    columns = offsets[:, 1, None] + torch.arange(width, device=images.device)
    column_index = columns[:, None, None, :].expand(
        -1, images.size(1), height, -1
    )
    images = images.gather(3, column_index)

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