MECHANISM: Boundary-seeking center-biased full-view pooling

HYPOTHESIS: A 1.625× center-offset weight will preserve 9,287 correct predictions while reducing cross-entropy below 0.2067078.

INTENDED_EDIT: Retain all ten evaluation views and the 1.225 logit scale, but weight the unshifted offset 1.625×.

EVIDENCE: Center weighting at 1.5× retained 9,287 correct and improved cross-entropy, whereas 1.75× lost one correct prediction; 1.625× is the informative midpoint between those outcomes.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return 1.225 * view_logits.mean(dim=(0, 1))
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.625 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.625
        return 1.225 * pooled_logits
>>>>>>> REPLACE