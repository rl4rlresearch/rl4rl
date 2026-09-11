MECHANISM: Batched test-time augmentation with interpolated temperature calibration

HYPOTHESIS: Temperature 0.912028 will preserve all 9,286 predictions while reducing cross-entropy below 0.19609205169677735, and vectorized view evaluation will prevent timeout.

INTENDED_EDIT: Evaluate the ten existing weighted views in two batched model passes while preserving their fusion order, then apply temperature 0.912028.

EVIDENCE: Temperatures 0.91, 0.912, and 0.91213 locate the cross-entropy minimum near 0.912028; batched TTA previously completed in 73.36 seconds, whereas unbatched calibration attempts timed out.

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

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        batch = images.shape[0]
        view_batch = torch.cat(views, dim=0)
        view_logits = self.classifier(self.features(view_batch)).reshape(
            5, batch, 10
        )
        flipped_logits = self.classifier(
            self.features(view_batch.flip(-1))
        ).reshape(5, batch, 10)

        ensemble = F.log_softmax(view_logits[0], dim=1) + math.log(2.0)
        for view_index in range(1, 5):
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits[view_index], dim=1)
            )
        for view_index in range(5):
            flipped_log_probs = F.log_softmax(
                flipped_logits[view_index], dim=1
            )
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912028
>>>>>>> REPLACE