MECHANISM: Arithmetic-probability flip ensembling

HYPOTHESIS: Averaging the original and flipped predictive probabilities will increase validation_correct above 9,359 by allowing a confidently correct view to outweigh an uncertain disagreeing view.

INTENDED_EDIT: Combine the two established evaluation views in probability space, convert the average back to logits, and retain the best-known calibration scale.

EVIDENCE: Equal-weight translated views reduced validation_correct to 9,348, while the prior probability-ensemble verification timed out without testing its accuracy; this retries the still-unmeasured aggregation change without introducing weaker geometric views.

<<<<<<< SEARCH
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        log_probabilities = torch.logaddexp(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(flipped_logits, dim=-1),
        ) - math.log(2.0)
        return 1.226016 * log_probabilities
>>>>>>> REPLACE