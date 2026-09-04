MECHANISM: Stronger confidence-weighted probability TTA

HYPOTHESIS: Increasing per-view sharpening from the verified 1.05× to 1.10× will exceed 9,206 correct predictions by giving decisive TTA views more influence, while 1.10× outer sharpening will improve cross-entropy without changing the ensemble argmax.

INTENDED_EDIT: Sharpen every TTA member by 1.10× before probability averaging and apply the verified 1.10× post-ensemble calibration.

EVIDENCE: Reference Design 2 verified that 1.05× per-view sharpening with 1.10× outer sharpening preserves 9,206 correct predictions and improves cross-entropy to 0.224459; repeated 1.15× outer-only attempts timed out, so changing the per-view factor directly tests a remaining route to higher correctness.

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
        probability_sum = F.softmax(logits * 1.10, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
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
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE