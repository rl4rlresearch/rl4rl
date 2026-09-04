MECHANISM: Vectorized temperature-calibrated test-time augmentation

HYPOTHESIS: Vectorizing the unchanged 12-weight TTA ensemble and applying temperature 1.03 will preserve all 9,286 argmax predictions, reduce validation cross-entropy below 0.197145, and avoid the recurrent verification timeouts.

INTENDED_EDIT: Evaluate the ten augmented views in two batched backbone calls instead of ten sequential calls, then divide the ensemble logits by 1.03.

EVIDENCE: The current design has the best observed correct count, while prior temperature-1.03 attempts timed out without testing accuracy; positive temperature scaling preserves argmax exactly, and batching the existing views reduces evaluation overhead without changing their weights.

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
        return ensemble - math.log(12.0)
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
        view_logits = self.classifier(
            self.features(view_batch)
        ).reshape(len(views), batch, 10)
        flipped_logits = self.classifier(
            self.features(view_batch.flip(-1))
        ).reshape(len(views), batch, 10)

        ensemble = F.log_softmax(view_logits[0], dim=1) + math.log(2.0)
        for view_index in range(1, len(views)):
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits[view_index], dim=1)
            )
        for view_index in range(len(views)):
            flipped_log_probs = F.log_softmax(
                flipped_logits[view_index], dim=1
            )
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 1.03
>>>>>>> REPLACE