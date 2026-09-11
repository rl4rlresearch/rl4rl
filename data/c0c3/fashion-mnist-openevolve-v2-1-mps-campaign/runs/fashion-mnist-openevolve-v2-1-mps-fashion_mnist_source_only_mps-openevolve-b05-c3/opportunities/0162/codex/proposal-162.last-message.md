MECHANISM: Center-biased terminal fine-tuning with qualified eight-view TTA

HYPOTHESIS: Increasing terminal centered-crop frequency from 1/5 to 1/3 will exceed 9,209 correct predictions by reversing the centered-exposure reduction that dropped correctness to 9,193.

INTENDED_EDIT: Restore the best verified eight-pass ensemble, then add a second centered outcome to the terminal crop sampler while retaining one outcome per cardinal shift.

EVIDENCE: Reference Design 1 achieved 9,209 correct with 1/5 centered terminal sampling; reducing centered frequency to 1/9 produced only 9,193 correct, directly motivating a center-biased terminal phase.

<<<<<<< SEARCH
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.5
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.5,
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

        return (probability_sum / 9.0).clamp_min(1e-8).log().mul_(1.10)
=======
        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
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
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.10)
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