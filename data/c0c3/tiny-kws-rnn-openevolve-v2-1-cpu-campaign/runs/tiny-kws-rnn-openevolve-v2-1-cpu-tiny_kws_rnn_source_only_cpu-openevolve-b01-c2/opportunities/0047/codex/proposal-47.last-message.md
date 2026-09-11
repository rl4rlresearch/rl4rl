MECHANISM: Relaxed confidence-gated penultimate exit

HYPOTHESIS: Lowering the qualified penultimate exit threshold from 0.95 to 0.90 will preserve validation accuracy at or above 85% while increasing early exits and reducing total inference MACs below 256,362,120.

INTENDED_EDIT: Adopt the qualified 22-frame schedule, avoid classifier MACs before the penultimate step, and exit examples with at least 0.90 posterior confidence after step 21.

EVIDENCE: The 0.95 confidence-gated design achieved 85.52% accuracy while reducing mean execution to 21.696 steps; unlike every tested unconditional 21-step schedule, it demonstrates that confidence-selective final-frame omission preserves qualification, motivating a measured threshold relaxation.

<<<<<<< SEARCH
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(3, available_frames - 5)) + [available_frames - 4]
=======
    def classify(
        self,
        state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        hidden, summary, maximum, count = state
        if count[0, 0].item() < 21.0:
            return hidden[:, 0, :8]
        mean_output = summary / count.clamp_min(1.0)
        features = torch.cat(
            (mean_output, maximum, hidden[:, 0, :]), dim=1
        )
        relative_logits = self.classifier(features)
        reference_logit = relative_logits.new_zeros(
            relative_logits.shape[0], 1
        )
        return torch.cat((relative_logits, reference_logit), dim=1)

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
        return (count == float(total_steps - 1)) & (confidence >= 0.90)
>>>>>>> REPLACE