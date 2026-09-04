MECHANISM: Uniform probability-space ten-view ensembling

HYPOTHESIS: Averaging predictive probabilities across all ten equally sampled training transformations will exceed 9,285 correct predictions by avoiding the current flip-pair geometric pooling, which can let one uncertain flipped view suppress an otherwise correct prediction.

INTENDED_EDIT: Replace hierarchical flip-logit/position-probability aggregation with a uniform probability mixture over all five translations and both horizontal orientations.

EVIDENCE: The strongest verified design achieves 9,285 correct using uniformly sampled five-position, two-orientation augmentation, but inference combines flips differently from positions; matching the ensemble to that uniform augmentation distribution is an isolated, computation-neutral test.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return torch.logsumexp(
            F.log_softmax(view_logits, dim=-1), dim=(0, 1)
        ) - math.log(view_logits.shape[0] * view_logits.shape[1])
>>>>>>> REPLACE