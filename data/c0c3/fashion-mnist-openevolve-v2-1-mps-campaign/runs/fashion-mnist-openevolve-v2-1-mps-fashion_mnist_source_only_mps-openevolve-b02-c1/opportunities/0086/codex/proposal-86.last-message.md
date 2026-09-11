MECHANISM: Two-pass weighted logit-space TTA consensus

HYPOTHESIS: Weighted logit averaging will exceed 9,286 correct predictions by suppressing isolated overconfident view errors, while evaluating all ten views in exactly two batched passes will avoid the prior timeout.

INTENDED_EDIT: Replace ten sequential evaluation passes and probability-space fusion with two five-view batched passes and 2:1 center-weighted logit averaging at temperature 0.912.

EVIDENCE: Center-weight tuning plateaued at 9,286 correct, and batched probability TTA completed in 71.8 seconds; the subsequent logit-space hypothesis timed out without testing its accuracy claim.

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
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
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.classifier(self.features(images))

        batch = images.shape[0]
        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        view_batch = torch.cat(views, dim=0)
        view_logits = self.classifier(self.features(view_batch)).reshape(
            5, batch, 10
        )
        flipped_logits = self.classifier(
            self.features(view_batch.flip(-1))
        ).reshape(5, batch, 10)
        ensemble = (
            2.0 * view_logits[0]
            + view_logits[1:].sum(dim=0)
            + 2.0 * flipped_logits[0]
            + flipped_logits[1:].sum(dim=0)
        )
        return ensemble / 12.0 / 0.912
>>>>>>> REPLACE