MECHANISM: Antithetic translation-orbit ensemble supervision

HYPOTHESIS: Extending the 9,229-correct differential-basis model from flip-pair supervision to symmetric translation-pair supervision will exceed 9,229 correct predictions by directly optimizing a four-view approximation of the weighted translation-and-flip ensemble used for validation.

INTENDED_EDIT: Restore the verified five-channel differential image basis, pair every sampled translation with its inverse offset, and jointly supervise individual views, flip-pair averages, and the complete four-view translation-orbit average.

EVIDENCE: Reference Design 3 reached 9,229 correct after flip-averaged loss improved the 9,224-correct differential-basis model; earlier weighted translation sampling and weighted logit evaluation also produced large gains, motivating the same objective-alignment strategy across symmetric translations.

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
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
    translation_weights = images.new_tensor(
=======
    height, width = images.shape[-2:]
    translation_weights = images.new_tensor(
>>>>>>> REPLACE

<<<<<<< SEARCH
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
=======
    offsets = torch.stack(
        (flat_offsets.div(5, rounding_mode="floor"), flat_offsets.remainder(5)),
        dim=1,
    )
    offsets = torch.cat((offsets, 4 - offsets), dim=0)
    images = torch.cat((images, images), dim=0)
    labels = torch.cat((labels, labels), dim=0)
    padded = F.pad(images, (2, 2, 2, 2), mode="replicate")

    rows = offsets[:, 0, None] + torch.arange(height, device=images.device)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images), paired_labels, label_smoothing=0.03
    )
=======
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    unflipped_logits, flipped_logits = paired_logits.chunk(2, dim=0)

    flip_pair_logits = 0.5 * (unflipped_logits + flipped_logits)
    first_pair_logits, inverse_pair_logits = flip_pair_logits.chunk(2, dim=0)
    base_labels = labels.chunk(2, dim=0)[0]

    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    flip_pair_loss = F.cross_entropy(
        flip_pair_logits, labels, label_smoothing=0.03
    )
    translation_orbit_loss = F.cross_entropy(
        0.5 * (first_pair_logits + inverse_pair_logits),
        base_labels,
        label_smoothing=0.03,
    )
    return (
        0.25 * individual_loss
        + 0.25 * flip_pair_loss
        + 0.50 * translation_orbit_loss
    )
>>>>>>> REPLACE