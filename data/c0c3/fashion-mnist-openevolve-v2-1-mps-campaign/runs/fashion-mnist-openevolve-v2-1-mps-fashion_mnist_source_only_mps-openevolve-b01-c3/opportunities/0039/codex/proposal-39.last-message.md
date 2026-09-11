MECHANISM: Transformation-matched probability ensembling with linear tail averaging

HYPOTHESIS: Uniformly averaging probabilities across all ten independently sampled training transformations will exceed 9,285 correct predictions by avoiding premature flip-logit fusion, while retaining the strongest verified final-10% linear recency average.

INTENDED_EDIT: Restore final-10% linearly weighted parameter averaging and replace hierarchical inference aggregation with a uniform probability mixture over every cardinal-shift and flip view.

EVIDENCE: Final-10% linear recency averaging achieved the best verified result of 9,285 correct. Training samples uniformly from ten position/flip transformations, but current inference first geometrically combines flip pairs; directly marginalizing all ten view probabilities better matches that training distribution without additional inference work.

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
            10, images.shape[0], 10
        )
        return torch.logsumexp(
            F.log_softmax(view_logits, dim=-1), dim=0
        ) - math.log(view_logits.shape[0])
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.95 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
        average_weight = 1.0 / optimizer.tail_average_count
=======
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE