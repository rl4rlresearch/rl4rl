MECHANISM: Two-stage TTA confidence sharpening

HYPOTHESIS: Retaining verified 1.05× per-view sharpening and increasing only the decision-preserving post-ensemble factor to 1.10× will preserve 9,206 correct predictions while reducing validation cross-entropy below 0.227083.

INTENDED_EDIT: Sharpen every TTA member by 1.05× before probability averaging, then sharpen the ensemble log-probabilities by 1.10×.

EVIDENCE: Per-view plus post-ensemble 1.05× sharpening preserved 9,206 correct and improved cross-entropy from 0.235889 to 0.227083; the remaining underconfidence motivates a conservative increase to the argmax-invariant outer factor.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            probability_sum.add_(F.softmax(self._forward_once(view), dim=1))
            probability_sum.add_(
                F.softmax(self._forward_once(view.flip(-1)), dim=1)
            )

        return (probability_sum / 10.0).clamp_min(1e-8).log()
=======
        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE