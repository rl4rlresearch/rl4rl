MECHANISM: Interior terminal center-frequency interpolation

HYPOTHESIS: A 30% terminal centered-crop rate will exceed 9,210 correct predictions by targeting the apparent optimum between 20% (9,209), 33.3% (9,210), and 42.9% (9,208), while retaining the best verified eight-view ensemble.

INTENDED_EDIT: Restore the qualified eight-pass centered/shifted TTA and replace terminal crop sampling with 12 centered and seven outcomes per cardinal shift out of 40 equally likely outcomes.

EVIDENCE: Correctness rose from 9,209 at 20% centered exposure to 9,210 at 33.3%, then fell to 9,208 at 42.9%; these observations motivate testing a balanced 30% interior setting without changing model capacity.

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
        offsets_y = (
            1 + (directions == 2).long() - (directions == 1).long()
        )
        offsets_x = (
            1 + (directions == 4).long() - (directions == 3).long()
        )
=======
        directions = torch.randint(
            0, 40, (images.shape[0],), device=images.device
        )
        offsets_y = (
            1
            + ((directions >= 7) & (directions < 14)).long()
            - (directions < 7).long()
        )
        offsets_x = (
            1
            + ((directions >= 21) & (directions < 28)).long()
            - ((directions >= 14) & (directions < 21)).long()
        )
>>>>>>> REPLACE