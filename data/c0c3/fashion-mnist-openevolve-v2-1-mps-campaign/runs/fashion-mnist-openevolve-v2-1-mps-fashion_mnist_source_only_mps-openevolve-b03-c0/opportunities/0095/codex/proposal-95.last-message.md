MECHANISM: Consensus-gated translation correction

HYPOTHESIS: Allowing the 30% blend to change predictions only when all four translated views agree with its alternative class will exceed 9,319 correct predictions.

INTENDED_EDIT: Retain the existing blend for argmax-preserving cases, while accepting prediction changes supported unanimously by all four translated views.

EVIDENCE: Translation blending consistently reduced cross-entropy, indicating complementary signal; the current argmax gate prevents testing whether its strongest, view-consistent corrections improve accuracy.

<<<<<<< SEARCH
            translated_logits = self.classifier(
                self.features(translated_images)
            ).reshape(4, images.shape[0], 10).mean(dim=0)
            blended_logits = 0.70 * logits + 0.30 * translated_logits
            preserves_argmax = blended_logits.argmax(dim=1).eq(
                logits.argmax(dim=1)
            )
            logits = torch.where(
                preserves_argmax.unsqueeze(1),
                blended_logits,
                logits,
            )
=======
            translated_view_logits = self.classifier(
                self.features(translated_images)
            ).reshape(4, images.shape[0], 10)
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
>>>>>>> REPLACE