MECHANISM: Native-orientation-biased flip ensemble

HYPOTHESIS: Weighting native-view logits at 0.55 and flipped-view logits at 0.45 will change borderline argmax decisions and exceed 9,330 correct predictions without affecting training time or parameter count.

INTENDED_EDIT: Replace equal evaluation-time flip fusion with a slight native-orientation bias while retaining the verified 1.184 calibration scale.

EVIDENCE: The current equal-weight ensemble already verifies 9,330 correct, while repeated temperature-only changes cannot alter argmax predictions; adjusting the relative view weights is the smallest zero-training-cost change that can improve the primary metric.

<<<<<<< SEARCH
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.5 * (logits + flipped_logits))
        return logits
=======
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 1.184 * (0.55 * logits + 0.45 * flipped_logits)
        return logits
>>>>>>> REPLACE