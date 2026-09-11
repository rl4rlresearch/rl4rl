MECHANISM: Expanded penultimate-frame confidence exit

HYPOTHESIS: Lowering the qualified penultimate-step exit threshold from 0.80 to 0.75 will retain validation_accuracy >= 0.85 while reducing total_inference_macs below 747,806,724.

INTENDED_EDIT: Restore the qualified affine-scale-only input normalization and intermediate-classification shortcut, then exit penultimate-step examples whose softmax confidence is at least 0.75.

EVIDENCE: Lowering the threshold from 0.85 to 0.80 retained 85.276% accuracy while reducing mean recurrent steps from 21.400 to 21.334 and total inference MACs by 2,411,208, motivating another incremental threshold reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
=======
        self.input_norm = nn.LayerNorm(20, bias=False)
>>>>>>> REPLACE

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
        return penultimate & (confidence >= 0.75)


def build_model() -> nn.Module:
>>>>>>> REPLACE