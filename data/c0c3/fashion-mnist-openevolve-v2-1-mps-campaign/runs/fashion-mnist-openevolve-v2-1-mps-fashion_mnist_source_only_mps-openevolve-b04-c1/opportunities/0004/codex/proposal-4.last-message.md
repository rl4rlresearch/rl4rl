MECHANISM: Pooling-phase translation ensemble

HYPOTHESIS: Averaging center and four one-pixel translations, each with its horizontal mirror, will exceed 9,212 correct predictions by reducing the max-pooling phase sensitivity that remains after the successful mirror ensemble.

INTENDED_EDIT: Expand evaluation-only augmentation from two mirror views to ten views spanning the original position and one-pixel shifts in four directions; leave training and parameter count unchanged.

EVIDENCE: Evaluation-time mirror averaging improved the same model from 9,194 to 9,212 correct without retraining, showing that view ensembling reduces prediction variance; small translations target an additional invariance relevant to the model’s two max-pooling operations.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble_logits = []
        for view in views:
            ensemble_logits.append(self._forward_view(view))
            ensemble_logits.append(
                self._forward_view(torch.flip(view, dims=(-1,)))
            )
        return torch.stack(ensemble_logits).mean(dim=0)
>>>>>>> REPLACE