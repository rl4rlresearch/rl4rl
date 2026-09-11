MECHANISM: Expanded two-stage confidence-gated temporal exit

HYPOTHESIS: Lowering the qualified two-steps-early threshold from 0.98 to 0.95 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 740,170,248.

INTENDED_EDIT: Enable the trained readout at step 20, exit there at confidence >= 0.95, and retain the qualified 0.7475 penultimate-step exit.

EVIDENCE: Reference Design 3 qualified at 85.153% with 740,170,248 MACs using thresholds 0.98 and 0.7475; earlier lowering of a confidence threshold from 0.98 to 0.95 preserved accuracy while substantially reducing recurrent execution.

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        hidden, summary, count = state
        if count[0, 0].item() < 20.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        return self.classifier(
            torch.cat((mean_output, hidden[:, 0, :-1]), dim=1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        del step
        _hidden, _summary, count = state
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return penultimate & (confidence >= 0.85)
=======
        del step
        _hidden, _summary, count = state
        two_steps_early = count[:, 0] == float(total_steps - 2)
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (
            (two_steps_early & (confidence >= 0.95))
            | (penultimate & (confidence >= 0.7475))
        )
>>>>>>> REPLACE