MECHANISM: Qualified confidence-weighted eight-view TTA and centered terminal sampling

HYPOTHESIS: Restoring the exact Reference Design 1 ensemble and one-third centered terminal sampler will increase validation_correct from 9,206 to 9,210 while achieving cross-entropy near 0.22289843.

INTENDED_EDIT: Replace equal-weight ten-view inference with the verified weighted eight-view ensemble using 1.10× per-view and 1.178× final sharpening, and increase terminal centered-crop probability from one-fifth to one-third.

EVIDENCE: Reference Design 1 used these exact settings and achieved the best verified result: 9,210 correct and 0.2228984337 cross-entropy, versus the current design’s 9,206 correct and 0.2270833004 cross-entropy.

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

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
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