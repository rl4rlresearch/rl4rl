MECHANISM: Two-stage confidence-boundary bisection

HYPOTHESIS: A 0.965 two-steps-early threshold will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 740,170,248 by admitting more safe step-20 exits than the qualified 0.98 design.

INTENDED_EDIT: Enable classification at recurrent step 20, exit there at confidence >= 0.965, and retain the qualified 0.7475 penultimate-step exit.

EVIDENCE: The 0.98 two-stage design qualified at 85.153% and 740,170,248 MACs, while 0.95 reduced MACs to 734,052,924 but missed accuracy at 84.908%; testing their midpoint directly probes the unresolved confidence boundary.

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
        return penultimate & (confidence >= 0.90)
=======
        two_steps_early = count[:, 0] == float(total_steps - 2)
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (
            (two_steps_early & (confidence >= 0.965))
            | (penultimate & (confidence >= 0.7475))
        )
>>>>>>> REPLACE