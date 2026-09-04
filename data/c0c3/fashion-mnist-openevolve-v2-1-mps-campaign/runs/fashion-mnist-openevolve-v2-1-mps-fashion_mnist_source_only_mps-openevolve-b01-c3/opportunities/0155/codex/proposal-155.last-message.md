MECHANISM: Consensus-directed flip-pair reweighting

HYPOTHESIS: Penalizing only the consensus-dissenting member of each flip-inconsistent pair will exceed 9,289 correct predictions by preserving reliable transformed evidence while suppressing the specific outlier view.

INTENDED_EDIT: Replace pair-wide penalties with a 20% penalty on dissenting individual views, preserving the successful 10% total pair-weight reduction when one member agrees with the initial ensemble.

EVIDENCE: A 10% pair-wide penalty achieved 9,289 correct, while agreement-conditioned calibration showed that ensemble consensus carries reliability information; this combines both signals without probability-space pooling.

<<<<<<< SEARCH
        offset_logits = view_logits.mean(dim=1)
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.85 + 0.15 * flip_consistency)
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_logits.argmax(dim=-1) == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
=======
        view_predictions = view_logits.argmax(dim=-1)
        base_weights = view_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).view(5, 1, 1)
        initial_logits = (
            base_weights.unsqueeze(-1) * view_logits
        ).sum(dim=(0, 1)) / (2.0 * base_weights.sum())
        initial_predictions = initial_logits.argmax(dim=-1)
        flip_inconsistency = (
            view_predictions[:, 0] != view_predictions[:, 1]
        ).unsqueeze(1)
        dissent = flip_inconsistency & (
            view_predictions != initial_predictions[None, None, :]
        )
        view_weights = base_weights * (1.0 - 0.20 * dissent.float())
        pooled_logits = (
            view_weights.unsqueeze(-1) * view_logits
        ).sum(dim=(0, 1)) / view_weights.sum(dim=(0, 1)).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
        view_agreement = (
            view_predictions == pooled_predictions[None, None, :]
        ).float().mean(dim=(0, 1))
>>>>>>> REPLACE