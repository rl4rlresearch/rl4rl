MECHANISM: Reduced tail averaging with batched probability-space TTA

HYPOTHESIS: Reducing tail-average interpolation from 0.5 to 0.25 will reverse the accuracy loss seen at 0.75 and achieve at least 9,287 correct predictions; two-pass batched TTA will let the previously timed-out test complete.

INTENDED_EDIT: Evaluate the ten unchanged weighted views in two batched passes and reduce final interpolation toward the late-checkpoint average to 25%.

EVIDENCE: Increasing tail interpolation to 0.75 reduced correct predictions from 9,286 to 9,276, motivating movement in the opposite direction; batched probability TTA previously completed in 71.8 seconds with all 9,286 predictions preserved.

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
        original_logits = self.classifier(
            self.features(torch.cat(views, dim=0))
        ).reshape(5, batch, 10)
        flipped_logits = self.classifier(
            self.features(torch.cat(tuple(view.flip(-1) for view in views), dim=0))
        ).reshape(5, batch, 10)
        log_probs = F.log_softmax(
            torch.cat((original_logits, flipped_logits), dim=0), dim=2
        )
        log_weights = log_probs.new_tensor(
            [math.log(2.0), 0.0, 0.0, 0.0, 0.0,
             math.log(2.0), 0.0, 0.0, 0.0, 0.0]
        ).view(10, 1, 1)
        ensemble = torch.logsumexp(log_probs + log_weights, dim=0)
        return (ensemble - math.log(12.0)) / 0.912
>>>>>>> REPLACE

<<<<<<< SEARCH
                    parameter.lerp_(average, 0.5)
=======
                    parameter.lerp_(average, 0.25)
>>>>>>> REPLACE