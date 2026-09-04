MECHANISM: Memory-bounded pairwise TTA with temperature calibration

HYPOTHESIS: Pairwise TTA batching will avoid the timeout of five-view batches while preserving the 9,286 predictions, and temperature 1.03 will lower cross-entropy without changing argmax classes.

INTENDED_EDIT: Evaluate the unchanged weighted 12-view ensemble in batches of at most two views, then divide its logits by 1.03.

EVIDENCE: The sequential ensemble achieved the best 9,286 correct, while the prior five-view vectorization timed out; smaller batches retain its computation with lower peak memory, and positive temperature scaling preserves argmax exactly.

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
        return ensemble - math.log(12.0)
=======
        ensemble = F.log_softmax(logits, dim=1) + math.log(2.0)
        remaining_views = views[1:] + tuple(view.flip(-1) for view in views)
        batch = images.shape[0]
        for start in range(0, len(remaining_views), 2):
            view_group = remaining_views[start : start + 2]
            grouped_images = torch.cat(view_group, dim=0)
            grouped_logits = self.classifier(self.features(grouped_images))
            for offset, view_logits in enumerate(
                grouped_logits.split(batch, dim=0)
            ):
                log_probs = F.log_softmax(view_logits, dim=1)
                if start + offset == 4:
                    log_probs = log_probs + math.log(2.0)
                ensemble = torch.logaddexp(ensemble, log_probs)
        return (ensemble - math.log(12.0)) / 1.03
>>>>>>> REPLACE