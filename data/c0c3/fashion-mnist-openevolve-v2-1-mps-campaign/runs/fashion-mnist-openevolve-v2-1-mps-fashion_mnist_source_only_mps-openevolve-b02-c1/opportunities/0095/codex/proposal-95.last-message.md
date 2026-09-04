MECHANISM: Final-iterate training with bounded batched probability TTA

HYPOTHESIS: Removing late-checkpoint averaging will reverse the accuracy drift observed when its weight increased and achieve at least 9,287 correct predictions, while bounded batched TTA preserves the proven ensemble within the time limit.

INTENDED_EDIT: Keep the final optimizer iterate instead of blending checkpoint averages, and evaluate the nine auxiliary views in two memory-bounded batched calls while preserving fusion order, weights, and temperature.

EVIDENCE: Increasing tail-average weight from 0.5 to 0.75 reduced validation-correct from 9,286 to 9,276, motivating movement fully toward the final iterate; batched probability TTA previously completed while retaining all 9,286 predictions.

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
        batch = images.shape[0]
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)

        shifted_images = torch.cat(views[1:], dim=0)
        shifted_logits = self.classifier(
            self.features(shifted_images)
        ).reshape(4, batch, 10)
        for view_logits in shifted_logits.unbind(0):
            ensemble = torch.logaddexp(
                ensemble, F.log_softmax(view_logits, dim=1)
            )

        flipped_images = torch.cat(
            tuple(view.flip(-1) for view in views), dim=0
        )
        flipped_logits = self.classifier(
            self.features(flipped_images)
        ).reshape(5, batch, 10)
        for view_index, view_logits in enumerate(flipped_logits.unbind(0)):
            flipped_log_probs = F.log_softmax(view_logits, dim=1)
            if view_index == 0:
                flipped_log_probs = flipped_log_probs + math.log(2.0)
            ensemble = torch.logaddexp(ensemble, flipped_log_probs)
        return (ensemble - math.log(12.0)) / 0.912
>>>>>>> REPLACE

<<<<<<< SEARCH
    update = step + 1
    in_tail = update >= int(0.8 * total_steps)
    sample_tail = in_tail and (update % 8 == 0 or update == total_steps)
    if sample_tail:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
=======
    return None
>>>>>>> REPLACE