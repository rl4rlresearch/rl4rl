MECHANISM: Single-batch weighted logit-space TTA consensus

HYPOTHESIS: Weighted logit averaging will exceed 9,286 correct predictions by suppressing isolated overconfident view errors, while processing all nine remaining views in one batched forward pass will avoid the timeouts that prevented testing this fusion rule.

INTENDED_EDIT: Reuse the centered-view logits, batch the other nine translated/flipped views into one model call, and combine logits with the training-matched 2:1 center weighting at temperature 0.912.

EVIDENCE: Probability-space weight changes plateaued at 9,286 correct, while both logit-space attempts timed out; batched probability TTA completed in 71.8 seconds, motivating more aggressive vectorization of the still-untested fusion rule.

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
        remaining_views = torch.cat(
            (*views[1:], *(view.flip(-1) for view in views)), dim=0
        )
        remaining_logits = self.classifier(
            self.features(remaining_views)
        ).split(batch, dim=0)

        ensemble = logits * 2.0
        for view_logits in remaining_logits[:4]:
            ensemble = ensemble + view_logits
        ensemble = ensemble + remaining_logits[4] * 2.0
        for view_logits in remaining_logits[5:]:
            ensemble = ensemble + view_logits
        return (ensemble / 12.0) / 0.912
>>>>>>> REPLACE