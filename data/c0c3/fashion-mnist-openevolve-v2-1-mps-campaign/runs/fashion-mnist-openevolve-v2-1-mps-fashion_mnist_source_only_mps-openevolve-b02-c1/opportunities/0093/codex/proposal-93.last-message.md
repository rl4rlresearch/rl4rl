MECHANISM: Symmetric overlapping terminal pooling with batched TTA

HYPOTHESIS: Covering all 7×7 terminal features with overlapping 3×3 pooling will achieve at least 9,287 validation-correct predictions, while batched probability-space TTA will complete within the verification limit.

INTENDED_EDIT: Replace the final lossy 2×2 pool with a 3×3 stride-2 pool and evaluate the nine additional TTA views in two batched forward passes without changing fusion weights or temperature.

EVIDENCE: The prior terminal-pooling test timed out without testing its accuracy claim; batched probability TTA previously completed in 71.8 seconds while preserving all 9,286 baseline predictions.

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        shifted_batches = self.classifier(
            self.features(torch.cat(views[1:], dim=0))
        ).chunk(len(views) - 1, dim=0)
        for view_logits in shifted_batches:
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )

        flipped_images = torch.cat(
            tuple(view.flip(-1) for view in views), dim=0
        )
        flipped_batches = self.classifier(
            self.features(flipped_images)
        ).chunk(len(views), dim=0)
        for view_index, flipped_logits in enumerate(flipped_batches):
            flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
>>>>>>> REPLACE