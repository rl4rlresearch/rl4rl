MECHANISM: Unsorted top-four saliency with argmax-invariant temperature calibration

HYPOTHESIS: Avoiding unnecessary top-four sorting will reduce runtime while preserving at least 9,322 correct predictions, and temperature 1.05 will lower cross-entropy without changing ensemble argmaxes.

INTENDED_EDIT: Compute the existing top-four descriptor with `sorted=False` and divide only the final evaluation logits by 1.05.

EVIDENCE: The current top-four design achieved the best count of 9,322 but finished near the time limit; the earlier unsorted attempt also removed the random pre-flip and changed dropout RNG, so it did not isolate this runtime optimization, while evaluation temperature remains prediction-invariant.

<<<<<<< SEARCH
        channel_salient = features.flatten(2).topk(4, dim=2).values
=======
        channel_salient = features.flatten(2).topk(
            4, dim=2, sorted=False
        ).values
>>>>>>> REPLACE

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        ensemble_logits = (
            torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
        )
        return ensemble_logits / 1.05
>>>>>>> REPLACE