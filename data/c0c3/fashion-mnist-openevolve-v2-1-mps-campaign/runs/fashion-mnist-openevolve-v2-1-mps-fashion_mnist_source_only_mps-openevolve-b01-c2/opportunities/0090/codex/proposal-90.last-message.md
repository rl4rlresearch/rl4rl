MECHANISM: Probability-space flip ensembling

HYPOTHESIS: Averaging per-view class probabilities instead of logits will exceed 9,330 correct predictions by allowing a confident correct view to rescue borderline disagreements, while retaining the verified 1.184 calibration scale.

INTENDED_EDIT: Replace evaluation-time geometric/logit fusion with an arithmetic probability mixture converted back to ten-class logits.

EVIDENCE: The current calibrated equal-logit ensemble verifies 9,330 correct, while temperature-only changes cannot alter argmax decisions and the native-orientation bias experiment did not finish; symmetric probability fusion is a distinct zero-training-cost way to change only borderline ensemble predictions.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            ensemble_probabilities = 0.5 * (
                F.softmax(logits, dim=1)
                + F.softmax(flipped_logits, dim=1)
            )
            logits = 1.184 * torch.log(
                ensemble_probabilities.clamp_min(1.0e-8)
            )
        return logits
>>>>>>> REPLACE