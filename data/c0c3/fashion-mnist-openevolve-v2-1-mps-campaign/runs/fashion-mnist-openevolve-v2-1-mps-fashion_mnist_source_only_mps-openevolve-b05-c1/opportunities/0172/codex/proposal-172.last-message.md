MECHANISM: TTA-consensus-conditioned logit calibration

HYPOTHESIS: Centered per-image temperature adjustments based on ten-view prediction agreement will retain all 9,360 predictions while lowering validation cross-entropy below 0.18585695190429688.

INTENDED_EDIT: Preserve the verified weighted logit ensemble, then slightly sharpen high-consensus examples and soften low-consensus examples with a strictly positive scale that cannot change argmax predictions.

EVIDENCE: Global scaling preserved 9,360 correct and improved cross-entropy, while anisotropic TTA weighting lost three correct; this motivates richer calibration without altering ensemble decisions.

<<<<<<< SEARCH
        center_weight = 1.81732177734375
        logits = center_weight * self._forward_once(views[0])
        logits = logits + center_weight * self._forward_once(
            views[0].flip(-1)
        )
        for view in views[1:]:
            logits = logits + self._forward_once(view)
            logits = logits + self._forward_once(view.flip(-1))
        logits = logits / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )
        return 1.0671112 * logits
=======
        center_weight = 1.81732177734375
        view_logits = [
            self._forward_once(views[0]),
            self._forward_once(views[0].flip(-1)),
        ]
        for view in views[1:]:
            view_logits.append(self._forward_once(view))
            view_logits.append(self._forward_once(view.flip(-1)))

        logits = center_weight * (
            view_logits[0] + view_logits[1]
        )
        for translated_logits in view_logits[2:]:
            logits = logits + translated_logits
        logits = logits / (
            2.0 * center_weight + 2.0 * (len(views) - 1)
        )

        predictions = torch.stack(view_logits).argmax(dim=2)
        ensemble_prediction = logits.argmax(dim=1)
        agreement = predictions.eq(
            ensemble_prediction.unsqueeze(0)
        ).float().mean(dim=0)
        agreement = agreement - agreement.mean()
        confidence_scale = torch.exp(0.04 * agreement)
        return 1.0671112 * confidence_scale[:, None] * logits
>>>>>>> REPLACE