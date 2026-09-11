MECHANISM: Qualified confidence-weighted TTA with argmax-invariant calibration

HYPOTHESIS: Restoring the verified 1.10× per-view ensemble will recover 9,210 correct predictions, while increasing only its final multiplier to 1.20× will lower cross-entropy below 0.2230063 without changing those predictions.

INTENDED_EDIT: Restore the qualified 1/3-centered terminal sampler and weighted eight-view TTA, then apply 1.20× post-ensemble sharpening.

EVIDENCE: Reference Design 3 achieved 9,210 correct with 0.2230063 cross-entropy at 1.10× per-view and 1.15× final sharpening; prior final-sharpening increases preserved argmax decisions and lowered cross-entropy, while 1.20× remains unresolved only because verification timed out.

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
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
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
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.20)
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