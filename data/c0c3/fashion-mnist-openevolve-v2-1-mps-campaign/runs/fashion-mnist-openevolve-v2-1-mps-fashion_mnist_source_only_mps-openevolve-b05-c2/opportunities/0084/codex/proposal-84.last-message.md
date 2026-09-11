MECHANISM: Locally optimized margin-adaptive calibration

HYPOTHESIS: A slope of 0.0170 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22206263122558595.

INTENDED_EDIT: Replace residual-form uniform sharpening with max-centered, batch-mean-centered margin scaling at slope 0.0170.

EVIDENCE: Slopes 0.01, 0.0175, and 0.02 produced cross-entropies 0.2221172874, 0.2220626312, and 0.2220722351 while preserving 9,192 correct predictions; local quadratic interpolation places the minimum near 0.01702.

<<<<<<< SEARCH
        return torch.add(
            normalized_logits,
            normalized_logits,
            alpha=0.09786707162857056,
        )
=======
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        confidence_scale = (10500.0 / 9564.0) + 0.017 * (
            margin - margin.mean()
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
        return normalized_logits * confidence_scale
>>>>>>> REPLACE