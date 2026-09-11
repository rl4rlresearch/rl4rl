MECHANISM: Disagreement-adaptive flip-pair weighting

HYPOTHESIS: Scaling the verified 10% pair-wide penalty up toward 25% only when flip predictions have large probability-space disagreement will exceed 9,289 correct predictions while avoiding the calibration cost of uniformly applying the stronger penalty.

INTENDED_EDIT: Restore the successful 10% minimum penalty and continuously increase it to at most 25% using total-variation disagreement between each offset’s original and flipped predictions.

EVIDENCE: Uniform 10% and 25% penalties both achieved 9,289 correct, but 25% worsened cross-entropy; meanwhile, penalizing individual dissenting views fell to 9,287. This motivates retaining proven pair-wide suppression while reserving stronger weighting for the most unreliable pairs.

<<<<<<< SEARCH
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (0.925 + 0.075 * flip_consistency)
=======
        flip_consistency = (
            view_logits[:, 0].argmax(dim=-1)
            == view_logits[:, 1].argmax(dim=-1)
        ).float()
        view_probabilities = view_logits.softmax(dim=-1)
        flip_disagreement = 0.5 * (
            view_probabilities[:, 0] - view_probabilities[:, 1]
        ).abs().sum(dim=-1)
        inconsistency_penalty = (
            0.10 + 0.15 * flip_disagreement
        ) * (1.0 - flip_consistency)
        base_weights = offset_logits.new_tensor(
            (1.5578022218542173, 1.0, 1.0, 1.0, 1.0)
        ).unsqueeze(1)
        offset_weights = base_weights * (1.0 - inconsistency_penalty)
>>>>>>> REPLACE