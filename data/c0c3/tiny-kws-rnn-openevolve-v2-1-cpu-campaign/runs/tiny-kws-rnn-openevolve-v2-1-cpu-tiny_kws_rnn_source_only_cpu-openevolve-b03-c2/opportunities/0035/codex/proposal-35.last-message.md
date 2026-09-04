MECHANISM: Expanded penultimate-frame confidence exit

HYPOTHESIS: Lowering the qualified early-exit threshold from 0.85 to 0.80 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 750,217,932.

INTENDED_EDIT: Add the qualified intermediate-classification shortcut and exit examples at the penultimate recurrent step when classifier confidence is at least 0.80.

EVIDENCE: Lowering the threshold from 0.90 to 0.85 preserved 85.399% accuracy while reducing mean recurrent steps from 21.480 to 21.400 and total inference MACs by 2,902,380, motivating another incremental threshold reduction.

<<<<<<< SEARCH
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
=======
        hidden, summary, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return schedule


def build_model() -> nn.Module:
=======
        return schedule

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        penultimate = count[:, 0] == float(total_steps - 1)
        confidence = logits.softmax(dim=1).amax(dim=1)
        return penultimate & (confidence >= 0.80)


def build_model() -> nn.Module:
>>>>>>> REPLACE