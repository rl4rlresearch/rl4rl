MECHANISM: Confidence-gated adaptive recurrent depth

HYPOTHESIS: Allowing predictions above 90% confidence to exit after 16 recurrent steps will preserve at least 85% validation accuracy while lowering mean execution below the qualified fixed 20-step model and therefore reducing total inference MACs below 620,185,660.

INTENDED_EDIT: Add a parameter-free early-exit rule to the qualified 22-step, batch-size-32 model; uncertain examples retain all 22 steps while confident examples stop after step 16 or later.

EVIDENCE: The current 22-step model reaches 85.89% accuracy, providing more margin than the barely qualifying 20-step model at 85.03%; the failed fixed 19-step thinning result shows that removing context uniformly is risky, motivating conditional computation that preserves full context for difficult clips.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
=======
    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step, total_steps
        _hidden, _summary, count = state
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        return (count[:, 0] >= 16.0) & (confidence >= 0.90)

    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 6:
            return list(range(available_frames))
        return list(range(4, available_frames - 6))
>>>>>>> REPLACE