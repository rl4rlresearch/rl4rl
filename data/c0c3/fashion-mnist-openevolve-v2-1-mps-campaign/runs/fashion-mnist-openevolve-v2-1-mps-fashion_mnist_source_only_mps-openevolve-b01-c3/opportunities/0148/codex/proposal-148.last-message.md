MECHANISM: Lower-dose flip-consistency-weighted logit ensembling

HYPOTHESIS: A 7.5% penalty on flip-inconsistent offset pairs will retain 9,289 correct predictions while reducing validation cross-entropy below 0.20669593772888184.

INTENDED_EDIT: Replace fixed offset pooling with per-image weights of 0.925 for flip-inconsistent pairs and 1.0 for consistent pairs, preserving center preference and agreement calibration.

EVIDENCE: Reducing the penalty from 15% to 12.5% to 10% retained 9,289 correct while successively lowering cross-entropy; 7.5% is the next dose toward the minimum effective penalty.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        pooled_logits = (
            1.5578022218542173 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578022218542173
=======
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.925 + 0.075 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
>>>>>>> REPLACE