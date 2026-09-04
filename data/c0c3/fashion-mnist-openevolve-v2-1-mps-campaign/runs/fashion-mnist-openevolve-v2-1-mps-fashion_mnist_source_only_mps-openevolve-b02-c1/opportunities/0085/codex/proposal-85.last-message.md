MECHANISM: Weighted logit-space TTA consensus

HYPOTHESIS: Averaging view logits instead of probabilities will exceed 9,286 correct predictions by suppressing isolated, overconfident view errors while preserving the training-matched 2:1 center-to-translation weighting.

INTENDED_EDIT: Vectorize the existing ten views into two additional model passes and replace probability-mixture fusion with weighted logit averaging at the proven 0.912 temperature.

EVIDENCE: Center weights 1.75 and 2.25 failed to improve the 9,286-correct baseline, indicating weight tuning has plateaued; vectorized evaluation completed in 71.8 seconds, making the fusion rule the next low-cost, argmax-changing variable.

<<<<<<< SEARCH
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
        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        batch = images.shape[0]
        translated_logits = self.classifier(
            self.features(torch.cat(views[1:], dim=0))
        ).reshape(4, batch, 10)
        flipped_logits = self.classifier(
            self.features(torch.cat([view.flip(-1) for view in views], dim=0))
        ).reshape(5, batch, 10)
        ensemble_logits = (
            2.0 * logits
            + translated_logits.sum(dim=0)
            + 2.0 * flipped_logits[0]
            + flipped_logits[1:].sum(dim=0)
        )
        return ensemble_logits / (12.0 * 0.912)
>>>>>>> REPLACE