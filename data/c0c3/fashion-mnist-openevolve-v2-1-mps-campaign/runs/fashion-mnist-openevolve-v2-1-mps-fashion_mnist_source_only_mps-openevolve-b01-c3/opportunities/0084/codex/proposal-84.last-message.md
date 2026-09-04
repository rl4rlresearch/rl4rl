MECHANISM: Stronger center-biased full-view logit pooling

HYPOTHESIS: Increasing the unshifted offset weight from 1.5× to 2× while retaining every shifted and flipped view will exceed 9,287 correct predictions by more strongly favoring validation-aligned evidence.

INTENDED_EDIT: Restore linear-recency tail averaging, pool the center offset at 2× weight, retain the best-verified 1.225 logit scale, and leave training and parameter count unchanged.

EVIDENCE: Center weighting at 1.5× preserved 9,287 correct and improved cross-entropy from 0.2067398 to 0.2067078; a stronger weight tests whether the supported direction was too modest to change accuracy.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return view_logits.mean(dim=(0, 1))
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            2.0 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 6.0
        return 1.225 * pooled_logits
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 0.5
=======
        recency_power = 1.0
>>>>>>> REPLACE