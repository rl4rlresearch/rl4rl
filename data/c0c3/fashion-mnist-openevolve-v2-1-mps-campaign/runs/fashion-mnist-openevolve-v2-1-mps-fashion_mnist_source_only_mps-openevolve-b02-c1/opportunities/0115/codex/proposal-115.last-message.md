MECHANISM: Agreement-adaptive robust logit TTA

HYPOTHESIS: Downweighting transformed views that disagree with the preliminary weighted consensus will suppress harmful TTA outliers and achieve at least 9,312 correct validation predictions.

INTENDED_EDIT: Retain all ten existing views and base weights, but replace fixed weighted-logit averaging with per-image agreement-weighted aggregation.

EVIDENCE: Fixed weighted-logit TTA improved the best result from 9,310 to 9,311 correct, showing aggregation affects borderline decisions; adaptive reliability weighting directly extends that successful change without altering training, parameters, or model-forward count.

<<<<<<< SEARCH
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.912)
=======
        logit_views = [logits]
        base_weights = [2.0]
        for view in views[1:]:
            logit_views.append(self.classifier(self.features(view)))
            base_weights.append(1.0)
        for view_index, view in enumerate(views):
            logit_views.append(self.classifier(self.features(view.flip(-1))))
            base_weights.append(2.0 if view_index == 0 else 1.0)

        stacked = torch.stack(logit_views)
        weights = logits.new_tensor(base_weights).view(-1, 1)
        consensus = (stacked * weights.unsqueeze(-1)).sum(0) / weights.sum()
        view_probabilities = F.softmax(stacked, dim=-1)
        consensus_probabilities = F.softmax(consensus, dim=-1).unsqueeze(0)
        disagreement = (
            view_probabilities - consensus_probabilities
        ).square().sum(dim=-1)
        adaptive_weights = weights * torch.exp(-4.0 * disagreement)
        ensemble = (
            stacked * adaptive_weights.unsqueeze(-1)
        ).sum(0) / adaptive_weights.sum(0, keepdim=False).unsqueeze(-1)
        return ensemble / 0.912
>>>>>>> REPLACE