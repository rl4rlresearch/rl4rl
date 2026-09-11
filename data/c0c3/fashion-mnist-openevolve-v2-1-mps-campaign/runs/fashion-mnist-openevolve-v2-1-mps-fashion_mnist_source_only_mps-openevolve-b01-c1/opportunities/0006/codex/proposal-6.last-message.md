MECHANISM: Training-distribution-matched translation-and-flip test-time ensemble

HYPOTHESIS: Averaging logits across all twenty-five two-pixel translations and their horizontal flips will exceed 9,055 correct predictions because training sampled the same five-by-five reflected translation distribution, while the successful current ensemble covers only its central nine translations.

INTENDED_EDIT: Expand evaluation-time augmentation from 18 views over one-pixel shifts to 50 views over two-pixel shifts, without changing training or parameter count.

EVIDENCE: Expanding flip-only evaluation to nine translations plus flips improved validation correct from 9,039 to 9,055; matching the full translation range used during training is the direct next test of that successful mechanism.

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        logits_sum = None
        for offset_y in range(3):
            for offset_x in range(3):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
        return logits_sum / 18.0
=======
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        logits_sum = None
        for offset_y in range(5):
            for offset_x in range(5):
                view = padded[
                    :,
                    :,
                    offset_y : offset_y + 28,
                    offset_x : offset_x + 28,
                ]
                views = torch.cat((view, view.flip(-1)), dim=0)
                logits = self._forward_once(views)
                original_logits, flipped_logits = logits.chunk(2, dim=0)
                view_logits = original_logits + flipped_logits
                if logits_sum is None:
                    logits_sum = view_logits
                else:
                    logits_sum = logits_sum + view_logits
        return logits_sum / 50.0
>>>>>>> REPLACE