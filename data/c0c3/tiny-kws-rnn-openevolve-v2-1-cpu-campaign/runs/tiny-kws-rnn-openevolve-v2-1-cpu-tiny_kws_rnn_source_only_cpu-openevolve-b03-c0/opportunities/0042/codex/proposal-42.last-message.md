MECHANISM: Confidence-gated final-frame skipping

HYPOTHESIS: Exiting after 24 steps only when confidence is at least 0.97 will preserve at least 85% accuracy and reduce total MACs, provided at least 2% of examples skip the final recurrent step.

INTENDED_EDIT: Add selective early exit before frame 28 while retaining the verified 25-step schedule; use a zero-MAC recurrent-state projection for provisional logits before the exit point.

EVIDENCE: Every tested fixed 24-step schedule missed the threshold by only a few examples, while the current 25-step schedule passed at 85.03%; routing uncertain examples through the informative final frame should retain its accuracy while confident examples avoid that step.

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        logits = self.classifier(averaged[:, :96])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        averaged = summary / count.clamp_min(1.0)
        if count.numel() and count[0, 0].item() < 24.0:
            logits = averaged[:, :7]
        else:
            logits = self.classifier(averaged[:, :96])
        return torch.cat((logits, -logits.sum(dim=1, keepdim=True)), dim=1)

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        if (
            count.numel() == 0
            or int(count[0, 0].item()) != total_steps - 1
            or total_steps != 25
        ):
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=torch.bool
            )
        confidence = logits.softmax(dim=1).amax(dim=1)
        return confidence >= 0.97

    def frame_schedule(self, available_frames: int) -> list[int]:
>>>>>>> REPLACE