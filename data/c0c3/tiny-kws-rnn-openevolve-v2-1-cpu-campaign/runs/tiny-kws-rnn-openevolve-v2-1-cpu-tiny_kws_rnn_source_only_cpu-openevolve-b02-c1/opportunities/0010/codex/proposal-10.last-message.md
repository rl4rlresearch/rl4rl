MECHANISM: Confidence-adaptive recurrent-depth control

HYPOTHESIS: Restoring all 32 frames but exiting high-confidence examples after 24–31 steps will retain at least 85% accuracy and reduce total MACs below 893,586,375 by reserving late recurrent updates for ambiguous recordings.

INTENDED_EDIT: Replace fixed first-frame trimming with full-frame processing and a progressively relaxed confidence-based exit policy beginning after 24 recurrent steps.

EVIDENCE: The 99-unit full-frame model reached 85.40%, while fixed 16- and 24-step schedules failed; this challenges the load-bearing assumption that every recording needs the same temporal budget by selectively shortening only confident examples.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 2:
            return list(range(available_frames))
        return list(range(1, available_frames))
=======
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
        seen = count[:, 0]
        eligible = seen >= 24.0
        progress = (
            (seen - 24.0) / float(max(total_steps - 24, 1))
        ).clamp(0.0, 1.0)
        confidence_threshold = 0.97 - 0.03 * progress
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        return eligible & (confidence >= confidence_threshold)
>>>>>>> REPLACE