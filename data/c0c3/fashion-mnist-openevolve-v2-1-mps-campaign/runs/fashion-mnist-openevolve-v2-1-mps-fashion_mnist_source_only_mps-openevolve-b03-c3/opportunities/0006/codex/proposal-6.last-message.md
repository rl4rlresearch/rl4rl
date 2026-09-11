MECHANISM: Lightly weighted radius-two translation-orbit ensembling

HYPOTHESIS: Extending the 9,042-correct 3×3 ensemble with lightly weighted two-pixel translations will exceed 9,042 correct predictions because training already samples the full ±2-pixel translation range.

INTENDED_EDIT: Evaluate horizontal-flip pairs across the complete 5×5 translation neighborhood, retaining most weight on the proven center and one-pixel views while assigning 10% total weight to the radius-two ring.

EVIDENCE: Reference Design 3 improved from 9,033 to 9,042 correct by adding previously omitted diagonal translations; this motivates testing the remaining translations represented by the training augmentation without changing the proven model or optimizer.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        height, width = images.shape[-2:]
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, :height, 1 : width + 1],
            padded[:, :, 2 : height + 2, 1 : width + 1],
            padded[:, :, 1 : height + 1, :width],
            padded[:, :, 1 : height + 1, 2 : width + 2],
        )

        logits = 0.5 * self._flip_average(images)
        for shifted in shifted_views:
            logits = logits + 0.125 * self._flip_average(shifted)
        return logits
=======
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