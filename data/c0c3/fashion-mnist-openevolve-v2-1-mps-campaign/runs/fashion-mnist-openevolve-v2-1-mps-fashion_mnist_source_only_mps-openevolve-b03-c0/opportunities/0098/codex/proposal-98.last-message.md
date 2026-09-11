MECHANISM: Unanimous-view confidence escalation

HYPOTHESIS: Raising translation weight to 40% only when all four translated views unanimously support a changed class will retain the existing corrections and exceed 9,321 correct predictions.

INTENDED_EDIT: Keep the argmax-preserving 30% blend, but use a 40% blend for prediction changes unanimously supported by all translated views.

EVIDENCE: The 30% blend with unanimous correction improved correctness from 9,319 to 9,321, whereas unrestricted three-of-four agreement fell to 9,319; this tests additional corrections without weakening unanimity.

<<<<<<< SEARCH
            translated_logits = translated_view_logits.mean(dim=0)
            blended_logits = 0.70 * logits + 0.30 * translated_logits
            base_predictions = logits.argmax(dim=1)
            blended_predictions = blended_logits.argmax(dim=1)
            preserves_argmax = blended_predictions.eq(base_predictions)
            translation_consensus = translated_view_logits.argmax(dim=2).eq(
                blended_predictions.unsqueeze(0)
            ).all(dim=0)
            use_blend = preserves_argmax | translation_consensus
            logits = torch.where(
                use_blend.unsqueeze(1),
                blended_logits,
                logits,
            )
=======
            translated_logits = translated_view_logits.mean(dim=0)
            blended_logits = 0.70 * logits + 0.30 * translated_logits
            consensus_logits = 0.60 * logits + 0.40 * translated_logits
            base_predictions = logits.argmax(dim=1)
            blended_predictions = blended_logits.argmax(dim=1)
            consensus_predictions = consensus_logits.argmax(dim=1)
            preserves_argmax = blended_predictions.eq(base_predictions)
            unanimous_correction = (
                consensus_predictions.ne(base_predictions)
                & translated_view_logits.argmax(dim=2).eq(
                    consensus_predictions.unsqueeze(0)
                ).all(dim=0)
            )
            logits = torch.where(
                unanimous_correction.unsqueeze(1),
                consensus_logits,
                torch.where(
                    preserves_argmax.unsqueeze(1),
                    blended_logits,
                    logits,
                ),
            )
>>>>>>> REPLACE