MECHANISM: Vectorized intermediate center-biased translation augmentation

HYPOTHESIS: A per-axis 1:3:1 crop distribution will exceed 9,221 correct predictions by interpolating between the best 1:2:1 distribution and the slightly worse 1:4:1 distribution, while vectorized inference will prevent the prior test from timing out.

INTENDED_EDIT: Change crop sampling from 1:2:1 to 1:3:1 and evaluate all 18 translated/flipped validation views in one forward pass.

EVIDENCE: The 1:2:1 distribution achieved 9,221 correct versus 9,220 for 1:4:1, making 1:3:1 the unmeasured intermediate; its prior verification timed out, while vectorizing an even larger 26-view ensemble completed successfully.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        views = (
            images,
            padded[..., :height, :width],
            padded[..., :height, 1 : width + 1],
            padded[..., :height, 2 : width + 2],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, :width],
            padded[..., 2 : height + 2, 1 : width + 1],
            padded[..., 2 : height + 2, 2 : width + 2],
        )
        log_probabilities = []
        for view in views:
            log_probabilities.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_probabilities.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        return torch.logsumexp(
            torch.stack(log_probabilities, dim=0), dim=0
        ) - math.log(len(log_probabilities))
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        height, width = images.shape[-2:]
        views = (
            images,
            padded[..., :height, :width],
            padded[..., :height, 1 : width + 1],
            padded[..., :height, 2 : width + 2],
            padded[..., 1 : height + 1, :width],
            padded[..., 1 : height + 1, 2 : width + 2],
            padded[..., 2 : height + 2, :width],
            padded[..., 2 : height + 2, 1 : width + 1],
            padded[..., 2 : height + 2, 2 : width + 2],
        )
        augmented = torch.cat(
            [
                transformed
                for view in views
                for transformed in (view, view.flip(-1))
            ],
            dim=0,
        )
        log_probabilities = F.log_softmax(
            self._forward_once(augmented), dim=1
        ).reshape(len(views) * 2, images.shape[0], -1)
        return torch.logsumexp(log_probabilities, dim=0) - math.log(
            len(views) * 2
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    offset_draw_y = torch.randint(0, 4, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 4, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 3).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 3).long()
=======
    offset_draw_y = torch.randint(0, 5, (batch,), device=images.device)
    offset_draw_x = torch.randint(0, 5, (batch,), device=images.device)
    offsets_y = (offset_draw_y >= 1).long() + (offset_draw_y == 4).long()
    offsets_x = (offset_draw_x >= 1).long() + (offset_draw_x == 4).long()
>>>>>>> REPLACE