MECHANISM: Late confidence-gated early exit

HYPOTHESIS: A 128-unit GRU using the qualified 28-frame schedule can retain at least 85% accuracy while a 0.99-confidence exit during the final four steps reduces exact validation MACs below the fixed 28-step design.

INTENDED_EDIT: Restore the qualified 28-frame uniform schedule and allow highly confident examples to stop during its final four recurrent steps.

EVIDENCE: The fixed 28-step, 128-unit design achieved 85.52% accuracy, while reducing recurrent width failed; confidence-gated execution targets dynamic step savings without removing recurrent capacity or temporal coverage from uncertain examples.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        steps = min(28, available_frames)
        return [
            round(index * (available_frames - 1) / (steps - 1))
            for index in range(steps)
        ]

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del state
        if step < total_steps - 4:
            return torch.zeros(
                logits.shape[0], device=logits.device, dtype=torch.bool
            )
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        return confidence >= 0.99
>>>>>>> REPLACE