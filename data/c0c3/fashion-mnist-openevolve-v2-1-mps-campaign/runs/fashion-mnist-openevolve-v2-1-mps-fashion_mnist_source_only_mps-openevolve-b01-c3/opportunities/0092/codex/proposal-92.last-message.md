MECHANISM: Quadratically calibrated center-biased ten-view pooling

HYPOTHESIS: A 1.225825 logit scale with the proven 1.546875× center weight will retain 9,287 correct predictions while reducing validation cross-entropy below 0.2067071121.

INTENDED_EDIT: Restore the best-verified center-biased pooling and replace its 1.225 scale with the estimated cross-entropy-minimizing scale of 1.225825.

EVIDENCE: The 1.546875× center weight achieved the best verified score; scales 1.125, 1.175, and 1.225 preserved identical argmax predictions while tracing a convex, diminishing cross-entropy curve whose quadratic minimum is approximately 1.225825.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return 1.125 * view_logits.mean(dim=(0, 1))
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.546875 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.546875
        return 1.225825 * pooled_logits
>>>>>>> REPLACE