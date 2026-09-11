MECHANISM: Consensus-regularized probability TTA

HYPOTHESIS: Blending 10% hard view-vote consensus into the verified probability ensemble will exceed 9,206 correct predictions by correcting close decisions dominated by a few confident dissenting views.

INTENDED_EDIT: Retain the verified 1.05× per-view and 1.10× outer sharpening, while accumulating top-class votes from the same ten TTA passes and blending them into the probability mean without adding parameters or model evaluations.

EVIDENCE: The unsharpened current design and Reference Design 2 both achieved exactly 9,206 correct despite per-view sharpening, showing confidence scaling alone did not improve decisions; Reference Design 2’s 0.224459 cross-entropy supports retaining its calibration while testing complementary view agreement.

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
        vote_sum = F.one_hot(
            logits.argmax(dim=1), num_classes=10
        ).to(dtype=logits.dtype)

        flipped_logits = self._forward_once(images.flip(-1))
        probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))
        vote_sum.add_(
            F.one_hot(
                flipped_logits.argmax(dim=1), num_classes=10
            ).to(dtype=logits.dtype)
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
            vote_sum.add_(
                F.one_hot(
                    view_logits.argmax(dim=1), num_classes=10
                ).to(dtype=logits.dtype)
            )
            vote_sum.add_(
                F.one_hot(
                    flipped_logits.argmax(dim=1), num_classes=10
                ).to(dtype=logits.dtype)
            )

        ensemble = (probability_sum + 0.10 * vote_sum) / 11.0
        return ensemble.clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE