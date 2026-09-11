MECHANISM: Confidence-gated penultimate-step early exit

HYPOTHESIS: Examples reaching at least 0.97 class probability after 27 of 28 scheduled steps can skip the final recurrence while preserving validation accuracy of at least 85% and reducing total inference MACs below 867,991,300.

INTENDED_EDIT: Add an `exit_mask` that stops only highly confident examples after the penultimate scheduled frame; uncertain examples retain all 28 steps.

EVIDENCE: Frames 4–31 qualified at 85.40%, while removing frame 31 for every example narrowly missed at 84.66%, motivating selective rather than universal omission of the final update.

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))


def build_model() -> nn.Module:
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        if available_frames <= 4:
            return list(range(available_frames))
        return list(range(4, available_frames))

    def exit_mask(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        _hidden, _summary, count = state
        penultimate_or_later = count[:, 0] >= float(total_steps - 1)
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        return penultimate_or_later & (confidence >= 0.97)


def build_model() -> nn.Module:
>>>>>>> REPLACE