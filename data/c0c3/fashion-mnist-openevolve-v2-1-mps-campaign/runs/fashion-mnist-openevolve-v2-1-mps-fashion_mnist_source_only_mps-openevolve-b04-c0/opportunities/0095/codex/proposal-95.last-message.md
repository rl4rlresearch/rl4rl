MECHANISM: Arithmetic-probability flip ensembling

HYPOTHESIS: Averaging original and flipped class probabilities instead of logits will increase validation_correct above 9,359 by allowing a confident correct view to outweigh an uncertain disagreeing view.

INTENDED_EDIT: Keep training and the best calibration factor unchanged, but combine the two evaluation views in probability space.

EVIDENCE: Float32 calibration refinements no longer changed predictions, while adding translated evaluation views reduced validation_correct to 9,348; this motivates changing how the two established original/flip views are aggregated without introducing weaker geometric views.

<<<<<<< SEARCH
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self.classifier(self.features(images.flip(-1)))
        log_probabilities = torch.logsumexp(
            torch.stack(
                (
                    F.log_softmax(logits, dim=-1),
                    F.log_softmax(flipped_logits, dim=-1),
                ),
                dim=0,
            ),
            dim=0,
        ) - math.log(2.0)
        return 1.226016 * log_probabilities
>>>>>>> REPLACE