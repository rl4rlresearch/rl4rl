MECHANISM: Stronger margin-adaptive logit calibration

HYPOTHESIS: Doubling the verified margin-adaptation slope from 0.01 to 0.02 will preserve all 9,192 argmax predictions and reduce validation cross-entropy below 0.22211728744506837.

INTENDED_EDIT: Replace uniform evaluation-time sharpening with positive, batch-mean-centered scaling that more strongly sharpens high-margin predictions and softens low-margin predictions.

EVIDENCE: Reference Design 3’s slope-0.01 margin adaptation preserved 9,192 correct predictions while improving cross-entropy from 0.22237433319091796 to 0.22211728744506837; testing a larger slope is the most direct probe of whether the observed gain continues.

<<<<<<< SEARCH
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        return normalized_logits * 1.0978671312332153
=======
        normalized_logits = logit_sum / 13.2124453125
        normalized_logits = normalized_logits - normalized_logits.amax(
            dim=1, keepdim=True
        )
        top_two = normalized_logits.topk(2, dim=1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        confidence_scale = (10500.0 / 9564.0) + 0.02 * (
            margin - margin.mean()
        )
        confidence_scale = confidence_scale.clamp(0.75, 1.5)
        return normalized_logits * confidence_scale
>>>>>>> REPLACE