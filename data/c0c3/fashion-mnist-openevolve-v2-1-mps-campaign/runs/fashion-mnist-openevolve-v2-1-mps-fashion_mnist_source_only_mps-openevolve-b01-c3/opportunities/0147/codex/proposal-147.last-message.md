MECHANISM: Lower-dose flip-consistency-weighted logit ensembling

HYPOTHESIS: A 10% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.2066986053466797.

INTENDED_EDIT: Replace fixed offset pooling with per-image flip-consistency weights of 0.90 for inconsistent pairs and 1.0 for consistent pairs, retaining the verified center preference and agreement calibration.

EVIDENCE: Reducing the penalty from 15% to 12.5% retained 9,289 correct and improved cross-entropy from 0.2067018039703369 to 0.2066986053466797, while increasing it to 25% worsened cross-entropy; testing 10% probes the minimum effective penalty.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.90 + 0.10 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE