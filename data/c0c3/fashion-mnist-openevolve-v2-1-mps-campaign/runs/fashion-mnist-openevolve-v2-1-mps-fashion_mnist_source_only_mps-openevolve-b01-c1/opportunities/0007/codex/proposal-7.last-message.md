MECHANISM: Center-weighted multi-radius test-time ensemble

HYPOTHESIS: Blending the 50-view ensemble with its central 18-view subset will exceed 9,063 correct predictions by retaining useful two-pixel views while reducing their influence on borderline examples.

INTENDED_EDIT: Average 75% of the existing five-by-five translation ensemble with 25% of the better-calibrated central three-by-three ensemble.

EVIDENCE: Uniform 50-view evaluation improved correct predictions from 9,055 to 9,063 but worsened cross-entropy from 0.26850 to 0.27379, indicating that outer translations add useful decisions yet should receive less weight.

<<<<<<< SEARCH
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
=======
        padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
        logits_sum = None
        central_logits_sum = None
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
                if 1 <= offset_y <= 3 and 1 <= offset_x <= 3:
                    if central_logits_sum is None:
                        central_logits_sum = view_logits
                    else:
                        central_logits_sum = central_logits_sum + view_logits
        uniform_logits = logits_sum / 50.0
        central_logits = central_logits_sum / 18.0
        return 0.75 * uniform_logits + 0.25 * central_logits
>>>>>>> REPLACE