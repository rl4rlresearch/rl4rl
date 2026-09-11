MECHANISM: Conservative hybrid geometric–arithmetic view pooling

HYPOTHESIS: Blending 12.5% reliability-weighted arithmetic probability pooling into the validated logit ensemble will exceed 9,325 correct predictions by damping isolated overconfident transformed views while preserving the stronger geometric-pooling signal.

INTENDED_EDIT: Keep training unchanged and replace evaluation logits with the log of an 87.5% logit-pooled and 12.5% probability-pooled mixture.

EVIDENCE: The weighted logit ensemble produced the best verified 9,325 correct; the prior 25% probability-blend attempt timed out rather than yielding negative accuracy evidence, motivating a more conservative blend with negligible added computation.

<<<<<<< SEARCH
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        offset_probabilities = view_logits.softmax(dim=-1).mean(dim=1)
        pooled_probabilities = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        hybrid_probabilities = (
            0.875 * pooled_logits.softmax(dim=-1)
            + 0.125 * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE