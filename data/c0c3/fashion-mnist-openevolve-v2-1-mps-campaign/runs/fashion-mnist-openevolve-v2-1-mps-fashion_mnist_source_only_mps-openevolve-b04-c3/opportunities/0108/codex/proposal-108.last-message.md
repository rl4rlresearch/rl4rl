MECHANISM: Verified power-mean flip ensembling with calibrated inference temperature

HYPOTHESIS: Using power 0.5831695556640625 and scaling inference logits by 1.03592 will preserve 9,243 correct predictions while reducing validation cross-entropy to approximately 0.208833005.

INTENDED_EDIT: Apply the best verified power-mean order and inference-only logit scale without changing training.

EVIDENCE: Reference Designs 2 and 3 both achieved the highest verified score, 9243.413622061835, with this exact inference configuration and otherwise identical code.

<<<<<<< SEARCH
            power = 0.583172607421875
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
=======
            power = 0.5831695556640625
            logits = 1.03592 * (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
>>>>>>> REPLACE