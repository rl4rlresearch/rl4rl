MECHANISM: Confidence-gated late recurrent exit with deferred learned-head evaluation

HYPOTHESIS: Evaluating the learned classifier only from step 28 onward and exiting examples whose top-two logit margin exceeds `3 + remaining_steps` will preserve at least 85% accuracy while reducing exact MACs through selective 28–31-step inference.

INTENDED_EDIT: Preserve the verified 100-unit GRU and training path exactly, use a zero-MAC recurrent-state proxy before step 28, and add conservative confidence-based exits during the final four possible recurrent steps.

EVIDENCE: The full-rate 100-unit GRU achieved 85.03%, while forcing every example to use 28 steps achieved only 83.93%; conditional late exit retains all 32 steps for uncertain examples while shortening only predictions already separated by a large margin.

<<<<<<< SEARCH
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def classify(
        self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _hidden, summary, count = state
        features = summary / count.clamp_min(1.0)
        if bool((count[0, 0] < 28.0).item()):
            return features[:, :8]
        return self.classifier(features)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        remaining = float(total_steps) - count[:, 0]
        if bool((remaining.max() > 4.0).item()):
            return torch.zeros_like(remaining, dtype=torch.bool)
        top_two = logits.topk(2, dim=1).values
        margin = top_two[:, 0] - top_two[:, 1]
        required_margin = 3.0 + remaining
        return (
            (remaining >= 1.0)
            & (remaining <= 4.0)
            & (margin >= required_margin)
        )
>>>>>>> REPLACE