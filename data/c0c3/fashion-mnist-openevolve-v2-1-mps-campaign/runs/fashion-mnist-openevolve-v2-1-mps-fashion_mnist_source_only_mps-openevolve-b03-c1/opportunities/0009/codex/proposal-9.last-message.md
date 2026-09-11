MECHANISM: Center-weighted matched test-time ensembling

HYPOTHESIS: Weighting the centered evaluation view 3:2 relative to each one-pixel cardinal translation will exceed 9,088 correct predictions by matching the successful triangular training-crop distribution.

INTENDED_EDIT: Change the ten-view test-time ensemble from equal averaging to triangular-distribution weights while retaining the same views and horizontal flips.

EVIDENCE: Center-weighted training translations improved accuracy from 9,082 to 9,088, while uniformly averaging the full translation support fell to 9,074; this motivates testing center bias at inference without changing training.

<<<<<<< SEARCH
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        logits = []
        for view in views:
            logits.append(self._forward_once(view))
            logits.append(self._forward_once(view.flip(-1)))
        return torch.stack(logits, dim=0).mean(dim=0)
=======
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        view_weights = (3.0, 2.0, 2.0, 2.0, 2.0)
        logits = []
        for view, weight in zip(views, view_weights):
            logits.append(self._forward_once(view) * weight)
            logits.append(self._forward_once(view.flip(-1)) * weight)
        return torch.stack(logits, dim=0).sum(dim=0) / (
            2.0 * sum(view_weights)
        )
>>>>>>> REPLACE