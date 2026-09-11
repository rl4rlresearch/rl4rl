MECHANISM: Conservative unanimous-view confidence escalation

HYPOTHESIS: A 35% translation blend used only for unanimous alternative-class corrections will preserve the 9,321-correct baseline and add at least one correct prediction.

INTENDED_EDIT: Keep the argmax-preserving blend at 30%, but evaluate unanimous prediction changes with a separate 35% translation blend.

EVIDENCE: The 30% unanimous gate improved correctness from 9,319 to 9,321, while relaxing consensus to three views reduced it to 9,319; a 35% unanimous-only blend cautiously expands corrections without weakening the successful agreement criterion.

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
            preserving_logits = 0.70 * logits + 0.30 * translated_logits
            correction_logits = 0.65 * logits + 0.35 * translated_logits
            base_predictions = logits.argmax(dim=1)
            preserving_predictions = preserving_logits.argmax(dim=1)
            correction_predictions = correction_logits.argmax(dim=1)
            preserves_argmax = preserving_predictions.eq(base_predictions)
            unanimous_correction = (
                correction_predictions.ne(base_predictions)
                & translated_view_logits.argmax(dim=2).eq(
                    correction_predictions.unsqueeze(0)
                ).all(dim=0)
            )
            logits = torch.where(
                unanimous_correction.unsqueeze(1),
                correction_logits,
                torch.where(
                    preserves_argmax.unsqueeze(1),
                    preserving_logits,
                    logits,
                ),
            )
>>>>>>> REPLACE