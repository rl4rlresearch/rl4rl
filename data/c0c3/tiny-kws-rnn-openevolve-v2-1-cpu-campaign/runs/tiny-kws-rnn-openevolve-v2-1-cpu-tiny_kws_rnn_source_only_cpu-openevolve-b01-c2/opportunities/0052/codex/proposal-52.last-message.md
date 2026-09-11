MECHANISM: Incrementally relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the qualified penultimate exit threshold from 0.79 to 0.77 will retain validation accuracy at or above 85% while reducing total inference MACs below 252,400,140.

INTENDED_EDIT: Avoid classifier MACs before step 21 and exit examples whose penultimate-step posterior confidence is at least 0.77, while preserving the qualified model and 22-frame schedule.

EVIDENCE: Thresholds from 0.95 through 0.79 all qualified; the 0.79 design achieved 85.28% accuracy and 252,400,140 MACs, leaving a modest accuracy margin for another measured 0.02 relaxation.

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
=======
        hidden, summary, maximum, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]


def build_model() -> nn.Module:
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 10)) + [
            available_frames - 8,
            available_frames - 6,
            available_frames - 4,
        ]

    def exit_mask(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        logits: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        del step
        count = state[3][:, 0]
        confidence = logits.softmax(dim=1).amax(dim=1)
        return (count == float(total_steps - 1)) & (confidence >= 0.77)


def build_model() -> nn.Module:
>>>>>>> REPLACE