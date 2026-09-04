MECHANISM: Two-stage confidence-gated temporal exit

HYPOTHESIS: Evaluating the trained readout two steps before completion and exiting only at confidence >= 0.98 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 746,601,120 by saving one additional recurrent step on enough highly confident examples to amortize the extra readout.

INTENDED_EDIT: Enable dense classification at recurrent step 20, add a conservative 0.98-confidence exit there, and retain the qualified 0.7475 penultimate-step exit.

EVIDENCE: The 0.98 penultimate exit preserved the no-exit model’s 85.276% accuracy exactly across 180 exits, while unconditional frame removal scored only 84.54%; this supports testing selective high-confidence removal of one additional frame now that threshold-only bisection below 0.7475 repeatedly failed.

<<<<<<< SEARCH
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
=======
        if count[0, 0].item() < 20.0:
            return hidden[:, 0, :8]
>>>>>>> REPLACE

<<<<<<< SEARCH
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return penultimate & (confidence >= 0.7475)
=======
        two_steps_early = count[:, 0] == float(total_steps - 2)
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (
            (two_steps_early & (confidence >= 0.98))
            | (penultimate & (confidence >= 0.7475))
        )
>>>>>>> REPLACE