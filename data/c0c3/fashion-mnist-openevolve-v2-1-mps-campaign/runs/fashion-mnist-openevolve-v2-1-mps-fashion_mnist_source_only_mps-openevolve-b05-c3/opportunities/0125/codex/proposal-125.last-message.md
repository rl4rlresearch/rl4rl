MECHANISM: Stronger confidence-weighted probability TTA

HYPOTHESIS: Increasing per-view sharpening from 1.05× to 1.10× will exceed 9,206 correct predictions by giving decisive TTA views more influence, while the argmax-invariant 1.10× outer calibration preserves cross-entropy gains.

INTENDED_EDIT: Sharpen every original, flipped, and shifted TTA member by 1.10× before probability averaging; leave training and outer calibration unchanged.

EVIDENCE: The verified 1.05× per-view design achieved 9,206 correct with 0.224459 cross-entropy. The only 1.10× per-view attempt timed out, so its decision-changing effect remains untested and supplies no contrary performance evidence.

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
>>>>>>> REPLACE