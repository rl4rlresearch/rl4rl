MECHANISM: Tempered power-mean flip ensembling

HYPOTHESIS: A temperature-2 interpolation between probability averaging and logit averaging will exceed 9,322 correct predictions by rewarding cross-orientation consensus without fully adopting the unverified geometric endpoint.

INTENDED_EDIT: Keep the reliable 9,320-correct model and training procedure unchanged, but replace arithmetic probability ensembling with a temperature-2 power mean during validation.

EVIDENCE: Arithmetic probability ensembling reached 9,320 correct, while ensemble-aware training reduced this to 9,307 and pure logit averaging timed out; an inference-only interpolation isolates ensemble geometry without adding training work.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
=======
        ensemble_temperature = 2.0
        return ensemble_temperature * (
            torch.logsumexp(
                log_probabilities / ensemble_temperature,
                dim=0,
            )
            - math.log(2.0)
        )
>>>>>>> REPLACE