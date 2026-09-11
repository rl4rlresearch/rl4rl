MECHANISM: Confidence-weighted probability TTA

HYPOTHESIS: Sharpening each view’s logits by 1.05 before probability averaging will exceed 9,206 correct predictions by favoring views with stronger class margins, while retaining the verified post-ensemble sharpening for tied-count cross-entropy gains.

INTENDED_EDIT: Apply 1.05× sharpening independently to every original, flipped, and shifted TTA member before averaging their probabilities.

EVIDENCE: Post-ensemble 1.05× sharpening preserved 9,206 correct predictions and reduced cross-entropy from 0.235889 to 0.230772, demonstrating underconfidence; moving the same signal inside the ensemble can additionally alter ambiguous decisions toward confident views.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )
=======
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for view in shifted_views:
            probability_sum.add_(F.softmax(self._forward_once(view), dim=1))
            probability_sum.add_(
                F.softmax(self._forward_once(view.flip(-1)), dim=1)
            )
=======
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))
>>>>>>> REPLACE