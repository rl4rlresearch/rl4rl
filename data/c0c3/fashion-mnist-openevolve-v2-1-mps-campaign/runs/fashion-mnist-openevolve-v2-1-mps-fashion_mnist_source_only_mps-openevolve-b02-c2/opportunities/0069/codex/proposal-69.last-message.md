MECHANISM: Loss-aligned geometric flip ensemble

HYPOTHESIS: Averaging orientation logits at validation will exceed 9,322 correct predictions by requiring cross-orientation class agreement consistent with the paired-view cross-entropy objective.

INTENDED_EDIT: Preserve the best top-four attention training design and replace arithmetic probability averaging with a faster geometric-probability ensemble implemented as mean logits.

EVIDENCE: The top-four model achieved 9,322 correct with independent paired-view cross-entropy, while adding an arithmetic-ensemble-aware loss reduced performance to 9,307; this motivates aligning inference with the successful individual-view log-loss rather than the harmful arithmetic objective.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE