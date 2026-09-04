MECHANISM: First-epoch learning-rate hold with second-epoch cosine consolidation

HYPOTHESIS: Holding the peak learning rate through the first 50,000 examples before cosine decay will exceed 9,322 correct predictions by providing more optimization under the fixed two-exposure budget, while probability-space flip ensembling will improve cross-entropy if correct counts tie.

INTENDED_EDIT: Delay cosine annealing until halfway through training and replace mean-logit validation ensembling with the verified arithmetic probability mixture.

EVIDENCE: Hard-maximum attention repeatedly completed near 75–78 seconds with 9,320 correct, whereas the two-prediction top-four improvement repeatedly risked timeout. Reference Designs 1 and 2 also showed that probability ensembling lowers hard-maximum cross-entropy without changing its correct count, motivating an optimization-only change that preserves the reliable architecture and runtime.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    progress = min(step / max(total_steps, 1), 1.0)
    anneal_progress = max(2.0 * progress - 1.0, 0.0)
    multiplier = 0.01 + 0.99 * 0.5 * (
        1.0 + math.cos(math.pi * anneal_progress)
    )
>>>>>>> REPLACE