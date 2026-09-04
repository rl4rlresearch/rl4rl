MECHANISM: Boundary-seeking per-view temperature interpolation

HYPOTHESIS: A 1.125× per-view multiplier will preserve the 9,210 correct predictions achieved at 1.10× while reducing validation cross-entropy below 0.2230063, locating a better point before the one-prediction regression at 1.15×.

INTENDED_EDIT: Restore the qualified 1/3-centered training sampler and eight-pass weighted TTA, use the midpoint 1.125× per-view sharpening, and retain 1.15× final calibration.

EVIDENCE: Reference Design 3 achieved 9,210 correct at 1.10× per-view sharpening, while 1.15× reduced cross-entropy slightly but lost one correct prediction; bisecting that interval is the most direct test for retaining correctness while improving the tie-breaker.

<<<<<<< SEARCH
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

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
        probability_sum = F.softmax(logits * 1.125, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.125,
                dim=1,
            ),
            alpha=0.375,
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.125, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.125, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.15)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE