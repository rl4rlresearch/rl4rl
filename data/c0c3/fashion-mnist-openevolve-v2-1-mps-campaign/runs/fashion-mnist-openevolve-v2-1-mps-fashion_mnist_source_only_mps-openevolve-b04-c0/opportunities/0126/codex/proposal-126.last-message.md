MECHANISM: Disagreement-targeted confidence correction

HYPOTHESIS: The adjacent ensemble weights differ on the single validation example responsible for the 9,359-to-9,358 regression; strongly favoring the accuracy-preserving prediction only on such disagreements will retain 9,359 correct while lowering cross-entropy below 0.1846570999145508.

INTENDED_EDIT: Keep refined logits for agreement cases, but add an 8-logit confidence boost to the stable ensemble’s predicted class when the two ensembles disagree.

EVIDENCE: The refined weighting lowered cross-entropy but lost exactly one correct prediction, while the existing argmax guard recovered that prediction without a measurable cross-entropy penalty, isolating disagreement cases as the most informative target.

<<<<<<< SEARCH
        unchanged_prediction = (
            refined_logits.argmax(dim=1) == stable_logits.argmax(dim=1)
        ).unsqueeze(1)
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            stable_logits,
        )
=======
        stable_prediction = stable_logits.argmax(dim=1, keepdim=True)
        unchanged_prediction = (
            refined_logits.argmax(dim=1, keepdim=True) == stable_prediction
        )
        confidence_boost = (~unchanged_prediction).to(refined_logits.dtype) * 8.0
        corrected_logits = refined_logits.scatter_add(
            1,
            stable_prediction,
            confidence_boost,
        )
        ensemble_logits = torch.where(
            unchanged_prediction,
            refined_logits,
            corrected_logits,
        )
>>>>>>> REPLACE