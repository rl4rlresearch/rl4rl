MECHANISM: Two-stage TTA confidence sharpening with split-rate EMA

HYPOTHESIS: The verified per-view sharpening and 0.0325 BatchNorm EMA will recover 9,206 correct predictions, while increasing the argmax-invariant outer sharpening to 1.10× will reduce cross-entropy below 0.227083.

INTENDED_EDIT: Apply 1.05× sharpening before averaging each TTA view, sharpen the ensemble log-probabilities by 1.10×, and use the validated 0.0325 terminal BatchNorm-buffer EMA rate.

EVIDENCE: Reference Design 3 achieved 9,206 correct and 0.227083 cross-entropy with 1.05× per-view and outer sharpening. Increasing only the outer factor preserves every ensemble argmax; its prior verification timed out and supplied no contrary performance evidence.

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

<<<<<<< SEARCH
                    ema_rate = 0.04 if is_buffer else 0.015
=======
                    ema_rate = 0.0325 if is_buffer else 0.015
>>>>>>> REPLACE